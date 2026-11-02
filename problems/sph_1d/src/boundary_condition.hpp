/*
 * Copyright (c) 2023 Alex Chen.
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

#pragma once

#include "data/fields.h"
#include "data/rng_states.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/domain_comm.h"
#include "grid_sph_1d.hpp"
#include "utils/nonown_ptr.hpp"
#include "utils/util_functions.h"
#include "systems/ptc_injector_new.h"
#include <memory>

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
class boundary_condition : public system_t {
 protected:
  const grid_curv_t<Conf>& m_grid;
  const domain_comm<Conf, ExecPolicy>* m_comm = nullptr;

  double m_w0 = 1.0;
  double m_freq = 5.0;
  double m_Bp = 1.0;
  int m_num_lambda = 4;
  bool m_moving_inject = false;
  double m_inject_lead = 0.05;
  double m_r_filled = 0.0;
  double m_upper = 0.0;
  bool m_moving_wall = false;
  nonown_ptr<particle_data_t> ptc;

  float m_q_e = 1.0;
  float m_kT = 1.0e-3;
  float m_rho0 = 0.0;
  int m_ppc = 0;

  nonown_ptr<vector_field<Conf>> E, B, E0, B0;
  ptc_injector_dynamic<Conf> m_injector;

 public:
  static std::string name() { return "boundary_condition"; }

  boundary_condition(const grid_curv_t<Conf>& grid,
                     const domain_comm<Conf, ExecPolicy>* comm = nullptr)
      : m_grid(grid), m_comm(comm), m_injector(grid) {}

  void init() override {
    sim_env().get_data("Edelta", E);
    sim_env().get_data("E0", E0);
    sim_env().get_data("Bdelta", B);
    sim_env().get_data("B0", B0);

    sim_env().params().get_value("w0", m_w0);
    sim_env().params().get_value("wave_freq", m_freq);
    sim_env().params().get_value("Bp", m_Bp);
    sim_env().params().get_value("num_lambda", m_num_lambda);

    sim_env().params().get_value("q_e", m_q_e);
    sim_env().params().get_value("kT", m_kT);
    sim_env().params().get_value("rho0", m_rho0);
    sim_env().params().get_value("ppc", m_ppc);

    sim_env().params().get_value("moving_inject", m_moving_inject);
    sim_env().params().get_value("inject_lead", m_inject_lead);
    sim_env().params().get_value("moving_wall", m_moving_wall);
    sim_env().get_data("particles", ptc);
    // std::vector<double> lower, size;
    // sim_env().params().get_value("lower", lower);
    // sim_env().params().get_value("size", size);
    m_r_filled = m_grid.lower[0];
    m_upper = m_grid.lower[0] + m_grid.sizes[0];
  }

  void update(double dt, uint32_t step) override {
    if (m_comm == nullptr || m_comm->domain_info().is_boundary[0]) {
      typedef typename Conf::idx_t idx_t;
      typedef typename Conf::value_t value_t;

      value_t time = sim_env().get_time();
      value_t Bp = m_Bp;
      value_t omega;
      value_t phase = 2.0 * M_PI * m_freq * time;
      if (phase < 2.0 * M_PI * m_num_lambda)
        omega = m_w0 * sin(phase);
      else
        omega = 0.0;

      ExecPolicy<Conf>::launch(
          [omega, Bp] LAMBDA(auto e, auto b, auto e0, auto b0) {
            auto& grid = ExecPolicy<Conf>::grid();
            auto ext = grid.extent();
              // For quantities that are not continuous across the surface
              // int n0 = grid.guard[0];
              ExecPolicy<Conf>::loop(0, grid.guard[0] + 1, [&] LAMBDA(auto n0) {
                auto idx = idx_t(index_t<1>(n0), ext);
                if (n0 < grid.guard[0]) {          // inside the star
                  e[0][idx] = 0.0;
                  b[1][idx] = 0.0;
                  b[2][idx] = 0.0;
                }
                value_t r = grid_sph_1d_t<Conf>::radius(grid.coord(0, n0, true));
                b[0][idx] = 0.0;                   // on and inside the surface
                e[1][idx] = 0.0;
                e[2][idx] = omega * Bp / r; // flipped the sign. Let's see what happens
              });
          },
          E, B, E0, B0);
      ExecPolicy<Conf>::sync();
    }
    
    // if (r_front - m_r_filled > 10 * m_grid.delta[0]) {
    if (m_moving_inject && m_r_filled < m_upper) {
      typedef typename Conf::value_t value_t;
      double r_front = std::min(m_upper, m_grid.lower[0] + sim_env().get_time() + m_inject_lead);
      if (r_front - m_r_filled > 10 * m_grid.delta[0] || (r_front == m_upper && m_r_filled < m_upper)) {
        value_t r_lo = m_r_filled, r_hi = r_front;
        value_t rho0 = m_rho0, kT = m_kT, q_e = m_q_e;
        int ppc = m_ppc;

        m_injector.inject_pairs(
          [r_lo, r_hi] LAMBDA(auto& pos, auto& grid, auto& ext) {
            value_t r = grid.coord(0, pos[0], false);
            return r > r_lo && r <= r_hi;
          },
          [ppc] LAMBDA(auto& pos, auto& grid, auto& ext) { return 2 * ppc; },
          [kT] LAMBDA(auto& x_global, rand_state& state, PtcType type) {
            return rng_maxwell_juttner_3d<value_t>(state, kT);
          },
          [rho0, q_e, ppc] LAMBDA(auto& x_global, PtcType type) {
            value_t r = grid_sph_1d_t<Conf>::radius(x_global[0]);
            return rho0 / q_e / ppc / r;
          });
      m_r_filled = r_front;
      }
    }
    if (m_moving_wall) {
      typedef typename Conf::value_t value_t;
      typedef typename Conf::idx_t idx_t;
      double D = 2.0 * m_inject_lead + 0.8 / m_freq;
      value_t r_wall = std::max<value_t>(m_grid.lower[0],
                                         m_grid.lower[0] + sim_env().get_time() - D);
        if (r_wall > m_grid.lower[0]) {
        auto num = ptc->number();
        ExecPolicy<Conf>::launch(
            [r_wall, num] LAMBDA(auto ptc) {
              auto& grid = ExecPolicy<Conf>::grid();
              auto ext = grid.extent();
              ExecPolicy<Conf>::loop(0, num, [&] LAMBDA(auto n) {
                uint32_t cell = ptc.cell[n];
                if (cell == empty_cell) return;
                auto pos = get_pos(idx_t(cell, ext), ext);
                value_t r = grid.template coord<0>(pos[0], ptc.x1[n]);
                if (r < r_wall) ptc.cell[n] = empty_cell;
              });
            },
            ptc);
        ExecPolicy<Conf>::sync();
      }
    }
  }
};

}  // namespace Aperture
