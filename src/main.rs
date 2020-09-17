#[macro_use]
extern crate log;

use clap::{App, Arg};
use heuristics::parser::parse_instance;
use heuristics::solver::{
  calculate_cmax, hill_climber, print_solution, priority, random_restart_hill_climber, sequential,
  simulated_annealing, tabu_search, verify_solution,
};
use std::fs;
use std::time::Duration;

fn main() {
  env_logger::init();

  let matches = App::new("heuristics")
    .version("1.0")
    .about("Heuristic solvers for the job shop scheduling problem")
    .arg(
      Arg::with_name("instance")
        .long("instance")
        .help("Instance file name")
        .takes_value(true)
        .required(true),
    )
    .arg(
      Arg::with_name("solver")
        .long("solver")
        .help("Solver to use")
        .possible_values(&[
          "hill-climber",
          "random-restart-hill-climber",
          "tabu-search",
          "simulated-annealing",
          "priority-sps",
          "priority-lps",
          "priority-spt",
          "priority-lpt",
          "priority-lwrm",
          "priority-mwrm",
          "sequential",
        ])
        .takes_value(true)
        .required(true),
    )
    .arg(
      Arg::with_name("timeout")
        .long("timeout")
        .help("Timeout (in s) after which to abort the search")
        .takes_value(true)
        .required(true),
    )
    .arg(
      Arg::with_name("seed")
        .long("seed")
        .help("Seed for rng")
        .takes_value(true)
        .required(true),
    )
    .arg(
      Arg::with_name("sa-start-acceptance-ratio")
        .long("sa-start-acceptance-ratio")
        .help("Start acceptance ratio parameter for simulated annealing")
        .takes_value(true)
        .required_if("solver", "simulated-annealing")
        .requires_if("simulated-annealing", "solver"),
    )
    .arg(
      Arg::with_name("sa-delta")
        .long("sa-delta")
        .help("Cooling parameter for simulated annealing")
        .takes_value(true)
        .required_if("solver", "simulated-annealing")
        .requires_if("simulated-annealing", "solver"),
    )
    .get_matches();

  let solver = matches.value_of("solver").expect("Missing solver");
  let file = matches.value_of("instance").expect("Missing instance file");
  let timeout = Duration::from_secs(
    matches
      .value_of("timeout")
      .and_then(|m| m.parse().ok())
      .expect("Invalid timeout"),
  );
  let seed: u64 = matches
    .value_of("seed")
    .and_then(|m| m.parse().ok())
    .expect("Invalid seed");

  let contents = fs::read_to_string(file).expect("Error reading file");
  let instance = parse_instance(&contents).expect("Error parsing file");

  let solution = match solver {
    "random-restart-hill-climber" => {
      let config = random_restart_hill_climber::Config {
        timeout: timeout,
        seed: seed,
      };
      random_restart_hill_climber::find_solution(&instance, &config).to_solution()
    }
    "tabu-search" => {
      let config = tabu_search::Config {
        timeout: timeout,
        seed: seed,
      };
      tabu_search::find_solution(&instance, &config).to_solution()
    }
    "simulated-annealing" => {
      let start_acceptance_ratio: f64 = matches
        .value_of("sa-start-acceptance-ratio")
        .and_then(|m| m.parse().ok())
        .expect("Invalid start acceptance ratio");
      let delta: f64 = matches
        .value_of("sa-delta")
        .and_then(|m| m.parse().ok())
        .expect("Invalid delta");
      let config = simulated_annealing::Config {
        timeout: timeout,
        seed: seed,
        start_acceptance_ratio: start_acceptance_ratio,
        delta: delta,
      };
      simulated_annealing::find_solution(&instance, &config).to_solution()
    }
    "hill-climber" => {
      let solution = priority::find_solution_sps(&instance);
      hill_climber::improve_solution(&instance, &solution).to_solution()
    }
    "priority-sps" => priority::find_solution_sps(&instance),
    "priority-lps" => priority::find_solution_lps(&instance),
    "priority-spt" => priority::find_solution_spt(&instance),
    "priority-lpt" => priority::find_solution_lpt(&instance),
    "priority-lwrm" => priority::find_solution_lwrm(&instance),
    "priority-mwrm" => priority::find_solution_mwrm(&instance),
    "sequential" => sequential::find_solution(&instance),
    _ => panic!("Solver not implemented"),
  };

  let cmax = calculate_cmax(&instance, &solution);
  verify_solution(&instance, &solution).expect("Verification failed");

  println!("{}", cmax);
  print_solution(&instance, &solution);
}
