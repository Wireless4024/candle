//! Methods for backpropagation of gradients.
use crate::op::{BinaryOp, Op, ReduceOp, UnaryOp};
use crate::{Error, Result, Tensor, TensorId};
use ahash::*;

thread_local! {
    pub static CANDLE_GRAD_DO_NOT_DETACH: bool = {
        match std::env::var("CANDLE_GRAD_DO_NOT_DETACH") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl Tensor {
    /// Return all the nodes that lead to this value in a topologically sorted vec, the first
    /// elements having dependencies on the latter ones, e.g. the first element if any is the
    /// argument.
    /// This assumes that the op graph is a DAG.
    pub fn sorted_nodes(&self) -> Vec<&Tensor> {
        // The vec of sorted nodes is passed as an owned value rather than a mutable reference
        // to get around some lifetime limitations.
        fn walk<'a>(
            node: &'a Tensor,
            nodes: Vec<&'a Tensor>,
            already_seen: &mut HashMap<TensorId, bool>,
        ) -> (bool, Vec<&'a Tensor>) {
            if let Some(&tg) = already_seen.get(&node.id()) {
                return (tg, nodes);
            }
            let mut track_grad = false;
            let mut nodes = if node.is_variable() {
                // Do not call recursively on the "leaf" nodes.
                track_grad = true;
                nodes
            } else if node.dtype().is_int() {
                nodes
            } else if let Some(op) = node.op() {
                match op {
                    Op::IndexAdd(t1, t2, t3, _)
                    | Op::Scatter(t1, t2, t3, _)
                    | Op::ScatterAdd(t1, t2, t3, _)
                    | Op::CustomOp3(t1, t2, t3, _)
                    | Op::WhereCond(t1, t2, t3) => {
                        let (tg, nodes) = walk(t1, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(t2, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(t3, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::Conv1D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::ConvTranspose1D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::Conv2D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::ConvTranspose2D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::CustomOp2(lhs, rhs, _)
                    | Op::Binary(lhs, rhs, _)
                    | Op::Gather(lhs, rhs, _)
                    | Op::IndexSelect(lhs, rhs, _)
                    | Op::Matmul(lhs, rhs)
                    | Op::SliceScatter0(lhs, rhs, _) => {
                        let (tg, nodes) = walk(lhs, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(rhs, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::Cat(args, _) => args.iter().fold(nodes, |nodes, arg| {
                        let (tg, nodes) = walk(arg, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }),
                    Op::Affine { arg, mul, .. } => {
                        if *mul == 0. {
                            nodes
                        } else {
                            let (tg, nodes) = walk(arg, nodes, already_seen);
                            track_grad |= tg;
                            nodes
                        }
                    }
                    Op::Unary(_node, UnaryOp::Ceil)
                    | Op::Unary(_node, UnaryOp::Floor)
                    | Op::Unary(_node, UnaryOp::Round)
                    | Op::Unary(_node, UnaryOp::Sign) => nodes,
                    Op::Reshape(node)
                    | Op::UpsampleNearest1D { arg: node, .. }
                    | Op::UpsampleNearest2D { arg: node, .. }
                    | Op::AvgPool2D { arg: node, .. }
                    | Op::MaxPool2D { arg: node, .. }
                    | Op::Copy(node)
                    | Op::Broadcast(node)
                    | Op::Cmp(node, _)
                    | Op::Reduce(node, ReduceOp::Min | ReduceOp::Sum | ReduceOp::Max, _)
                    | Op::ToDevice(node)
                    | Op::Transpose(node, _, _)
                    | Op::Permute(node, _)
                    | Op::Narrow(node, _, _, _)
                    | Op::Unary(node, _)
                    | Op::Elu(node, _)
                    | Op::Powf(node, _)
                    | Op::CustomOp1(node, _) => {
                        let (tg, nodes) = walk(node, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::ToDType(node) => {
                        if node.dtype().is_float() {
                            let (tg, nodes) = walk(node, nodes, already_seen);
                            track_grad |= tg;
                            nodes
                        } else {
                            nodes
                        }
                    }
                    Op::Reduce(_, ReduceOp::ArgMin | ReduceOp::ArgMax, _) => nodes,
                }
            } else {
                nodes
            };
            already_seen.insert(node.id(), track_grad);
            if track_grad {
                nodes.push(node);
            }
            (track_grad, nodes)
        }
        let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::new());
        nodes.reverse();
        nodes
    }

    #[inline]
    pub fn backward(&self) -> Result<GradStore> {
        self.backward2(!CANDLE_GRAD_DO_NOT_DETACH.with(|b| *b))
    }
}

/// A store for gradients, associating a tensor id to the corresponding gradient tensor, used for back propagation.
#[derive(Debug, Default)]
pub struct GradStore(HashMap<TensorId, Tensor>);

impl GradStore {
    /// Create a new gradient store
    pub(crate) fn new() -> Self {
        GradStore(HashMap::new())
    }

    /// Get the gradient tensor corresponding to the given tensor id
    pub fn get_id(&self, id: TensorId) -> Option<&Tensor> {
        self.0.get(&id)
    }

    /// Get the gradient tensor associated with the given tensor
    pub fn get(&self, tensor: &Tensor) -> Option<&Tensor> {
        self.0.get(&tensor.id())
    }

    /// Remove the gradient tensor associated with the given tensor, returning it if it exists
    pub fn remove(&mut self, tensor: &Tensor) -> Option<Tensor> {
        self.0.remove(&tensor.id())
    }

    /// Insert a gradient tensor associated with the given tensor, returning the previous gradient tensor if it existed
    pub fn insert(&mut self, tensor: &Tensor, grad: Tensor) -> Option<Tensor> {
        self.0.insert(tensor.id(), grad)
    }

    /// Get the gradient tensor associated with the given tensor, or, if it does not exist,
    /// insert a tensor of zeroes, with the same shape and type as the given tensors and return it
    pub(crate) fn or_insert(&mut self, tensor: &Tensor) -> Result<&mut Tensor> {
        use std::collections::hash_map::Entry;
        let grad = match self.0.entry(tensor.id()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let grad = tensor.zeros_like()?;
                entry.insert(grad)
            }
        };
        Ok(grad)
    }

    /// Get the tensor ids of the stored gradient tensors
    pub fn get_ids(&self) -> impl Iterator<Item = &TensorId> {
        self.0.keys()
    }
}
