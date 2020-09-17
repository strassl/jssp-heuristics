use ndarray::Array1;

pub type Machine = usize;
pub type Duration = u32;
pub type Time = u32;

pub type Op = [usize; 2];

pub type OpId = usize;

pub type Edge = (OpId, OpId);

#[derive(Debug, Clone)]
pub struct Instance {
  pub n_machines: usize,
  pub n_jobs: usize,

  pub durations: Array1<Duration>,
  pub machines: Array1<Machine>,
}

#[derive(Debug, Clone)]
pub struct Solution {
  pub start_times: Array1<Time>,
}

impl Instance {
  pub fn ops(&self) -> Vec<Op> {
    let mut nodes = Vec::new();

    for j in 0..self.n_jobs {
      for o in 0..self.n_machines {
        nodes.push([j, o]);
      }
    }

    return nodes;
  }

  pub fn op_ids(&self) -> Vec<OpId> {
    return self.ops().into_iter().map(|op| self.op_to_id(op)).collect();
  }

  pub fn op_to_id(&self, op: Op) -> OpId {
    let [j, o] = op;
    return j * self.n_machines + o;
  }

  pub fn op_from_id(&self, id: OpId) -> Op {
    let j = id / self.n_machines;
    let o = id % self.n_machines;
    return [j, o];
  }

  pub fn shape(&self) -> (usize, usize) {
    return (self.n_jobs, self.n_machines);
  }

  pub fn n_ops(&self) -> usize {
    return self.n_jobs * self.n_machines;
  }
}
