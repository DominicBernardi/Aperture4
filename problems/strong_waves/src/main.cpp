/*
 * Copyright (c) 2022 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/compute_moments.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_cartesian.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater.h"
#include "utils/util_functions.h"
#include <iostream>

#include "boundary_condition.hpp"

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<1> Conf;
  auto &env = sim_environment::instance(&argc, &argv, true);
  typedef typename Conf::value_t value_t;

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_t<Conf> grid(comm);
  auto pusher = env.register_system<
      ptc_updater<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid,
                                                                      &comm);
  auto solver = env.register_system<
      field_solver<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid,
                                                                       &comm);
  // Registered right after the field solver so that it overwrites the guard
  // cells with the driven wave after every field update
  auto bc = env.register_system<boundary_condition<Conf, exec_policy_dynamic>>(
      grid, &comm);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto moments =
      env.register_system<compute_moments<Conf, exec_policy_dynamic>>(grid);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);

  env.init();

  // Background magnetic field, in the x-y plane at an angle theta_B (in
  // degrees) from the x axis, which is also the direction of the wave vector
  value_t Bp = 0.0, theta_B = 90.0;
  env.params().get_value("Bp", Bp);
  env.params().get_value("theta_B", theta_B);
  value_t theta = theta_B * M_PI / 180.0;

  vector_field<Conf> *B0, *Btotal;
  env.get_data("B0", &B0);
  env.get_data("B", &Btotal);
  B0->set_values(
      0, [Bp, theta](auto x, auto y, auto z) { return Bp * math::cos(theta); });
  B0->set_values(
      1, [Bp, theta](auto x, auto y, auto z) { return Bp * math::sin(theta); });
  // The field solver only folds B0 into the total B at the end of its first
  // update, but the pusher runs before it, so set the total B here too
  Btotal->copy_from(*B0);

  // Prepare initial conditions
  value_t rho_0 = 1.0, q_e = 1.0, kT = 1.0e-4;
  int ppc = 10;
  bool test_particles = false;
  env.params().get_value("rho_0", rho_0);
  env.params().get_value("q_e", q_e);
  env.params().get_value("ppc", ppc);
  env.params().get_value("kT", kT);
  env.params().get_value("test_particles", test_particles);

  // Test particles feel the fields but deposit no current, so the fields
  // evolve as in vacuum
  uint32_t flag = 0;
  if (test_particles) flag = flag_or(PtcFlag::test_particle);

  // Uniform, neutral, cold pair plasma filling the box. Unlike two_stream,
  // both electrons and positrons carry weight so the plasma is quasi-neutral.
  ptc_injector_dynamic<Conf> injector(grid);
  injector.inject_pairs(
      // Inject in every cell
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      // Number of particles per cell, including both species
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * ppc; },
      // Momentum distribution: a non-drifting Maxwell-Juttner at temperature kT
      [kT] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        return rng_maxwell_juttner_3d(state, kT);
      },
      // Particle weight
      [rho_0, ppc, q_e] LAMBDA(auto &x_global, PtcType type) {
        return rho_0 / ppc / q_e;
      },
      flag);

  env.run();
  return 0;
}
