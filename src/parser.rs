use crate::data::Instance;
use ndarray::Array1;
use std::error::Error;

pub fn parse_instance(instance: &String) -> Result<Instance, Box<dyn Error>> {
  let mut lines: Vec<&str> = instance.lines().collect();

  let prelude = lines.drain(0..1).next().ok_or("Prelude missing")?;
  let prelude_items: Vec<&str> = prelude.split_whitespace().collect();
  let n_jobs = prelude_items.get(0).ok_or("n_jobs missing")?.parse()?;
  let n_machines = prelude_items.get(1).ok_or("n_machines missing")?.parse()?;

  let mut instance = Instance {
    n_jobs: n_jobs,
    n_machines: n_machines,
    durations: Array1::<u32>::from_elem(n_jobs * n_machines, 0),
    machines: Array1::<usize>::from_elem(n_jobs * n_machines, 0),
  };

  for (job, line) in lines.iter().enumerate() {
    let items: Vec<&str> = line.split_whitespace().collect();
    for i in (0..items.len()).step_by(2) {
      let machine: usize = items.get(i).ok_or("Machine missing")?.parse()?;
      let duration: u32 = items.get(i + 1).ok_or("Duration missing")?.parse()?;

      let o = i / 2;
      let op = instance.op_to_id([job, o]);
      instance.durations[op] = duration;
      instance.machines[op] = machine;
    }
  }

  Ok(instance)
}
