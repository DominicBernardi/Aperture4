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

#include "core/multi_array_exp.hpp"
#include "core/ndsubset.hpp"
#include "core/ndsubset_dev.hpp"
#include "field_solver_sph_1d.h"
#include "systems/helpers/finite_diff_helper.hpp"
#include "systems/field_solver_base_impl.hpp"
#include "utils/double_buffer.h"

namespace Aperture {

namespace {

const int diff_order = 2;

template <int Dim>
using fd = finite_diff<Dim, diff_order>;

template <typename Conf, template <class> class ExecPolicy>
void
damping_boundary(vector_field<Conf>& e, vector_field<Conf>& b,
                 int damping_length, typename Conf::value_t damping_coef) {
  typedef typename Conf::idx_t idx_t;
  typedef typename Conf::value_t value_t;
  // kernel_launch(
  ExecPolicy<Conf>::launch(
      [damping_length, damping_coef] LAMBDA(auto e, auto b) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        // for (auto n1 : grid_stride_range(0, grid.dims[1])) {
        ExecPolicy<Conf>::loop(0, damping_length - 1, [&] LAMBDA(auto i) {
          int n0 = grid.dims[0] - damping_length + i;
          auto idx = idx_t(index_t<1>(n0), ext);
          value_t lambda =
              1.0f - damping_coef * cube((value_t)i / (damping_length - 1));
          e[0][idx] *= lambda;
          e[1][idx] *= lambda;
          e[2][idx] *= lambda;
          b[1][idx] *= lambda;
          b[2][idx] *= lambda;
        });
      },
      e, b);
  ExecPolicy<Conf>::sync();
}

template <typename VecType, typename Idx_t>
HD_INLINE auto circ1_1d(const VecType& f, const VecType& dl, const Idx_t& idx,
                        const Idx_t& idx_px) {
  return f[2][idx] * dl[2][idx] - f[2][idx_px] * dl[2][idx_px];
}

template <typename VecType, typename Idx_t>
HD_INLINE auto circ2_1d(const VecType& f, const VecType& dl,
                        const Idx_t& idx_mx, const Idx_t& idx_px) {
  return f[1][idx_px] * dl[1][idx_px] - f[1][idx_mx] * dl[1][idx_mx];
}

}  // namespace

template <typename Conf, template <class> class ExecPolicy>
field_solver<Conf, ExecPolicy, coord_policy_spherical_1d>::field_solver(
    const grid_sph_1d_t<Conf>& grid, const domain_comm<Conf, ExecPolicy>* comm)
    : field_solver_base<Conf>(grid), m_grid_sph(grid), m_comm(comm) {
  ExecPolicy<Conf>::set_grid(this->m_grid);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_spherical_1d>::init() {
  field_solver_base<Conf>::init();

  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("damping_coef", m_damping_coef);

//   if (this->m_use_implicit) {
//     auto type = ExecPolicy<Conf>::data_mem_type();
//     m_tmp_b1 = std::make_unique<vector_field<Conf>>(
//         this->m_grid, field_type::face_centered, type);
//     m_tmp_b2 = std::make_unique<vector_field<Conf>>(
//         this->m_grid, field_type::face_centered, type);
//     m_bnew = std::make_unique<vector_field<Conf>>(
//         this->m_grid, field_type::face_centered, type);
//   }
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy,
             coord_policy_spherical_1d>::register_data_components() {
  auto type = ExecPolicy<Conf>::data_mem_type();

  field_solver_base<Conf>::register_data_components_impl(type);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_spherical_1d>::update_explicit(
    double dt, double time) {
  if (time < TINY)
    compute_b_update_explicit(*(this->B), *(this->E), 0.5 * dt);
  else
    compute_b_update_explicit(*(this->B), *(this->E), dt);

  // Communicate B guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->B));

  compute_e_update_explicit(*(this->E), *(this->B), *(this->J), dt);

  // Communicate E guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));

  // apply coordinate boundary condition
  if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[1]) {
    damping_boundary<Conf, ExecPolicy>(*(this->E), *(this->B), m_damping_length,
                                       m_damping_coef);
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_spherical_1d>::compute_divs_e_b() {
  // vec_t<bool, Conf::dim * 2> is_boundary;
  // is_boundary.set(true);
  // if (m_comm != nullptr) {
  //   is_boundary = m_comm->domain_info().is_boundary;
  // }
  // ExecPolicy<Conf>::launch(
  //     [is_boundary] LAMBDA(auto divE, auto divB, auto e, auto b, auto gp) {
  //       auto& grid = ExecPolicy<Conf>::grid();
  //       auto ext = grid.extent();

  //       ExecPolicy<Conf>::loop(
  //           Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
  //             auto pos = get_pos(idx, ext);
  //             if (grid.is_in_bound(pos)) {
  //               auto idx_mx = idx.dec_x();
  //               auto idx_my = idx.dec_y();

  //               divE[idx] = (e[0][idx] * gp.Ae[0][idx] -
  //                            e[0][idx_mx] * gp.Ae[0][idx_mx] +
  //                            e[1][idx] * gp.Ae[1][idx] -
  //                            e[1][idx_my] * gp.Ae[1][idx_my]) /
  //                           (gp.dV[idx] * grid.delta[0] * grid.delta[1]);
  //               auto idx_px = idx.inc_x();
  //               auto idx_py = idx.inc_y();

  //               divB[idx] = (b[0][idx_px] * gp.Ab[0][idx_px] -
  //                            b[0][idx] * gp.Ab[0][idx] +
  //                            b[1][idx_py] * gp.Ab[1][idx_py] -
  //                            b[1][idx] * gp.Ab[1][idx]) /
  //                           (gp.dV[idx] * grid.delta[0] * grid.delta[1]);

  //               if (is_boundary[0] && pos[0] == grid.guard[0])
  //                 divE[idx] = divB[idx] = 0.0f;
  //               if (is_boundary[1] &&
  //                   pos[0] == grid.dims[0] - grid.guard[0] - 1)
  //                 divE[idx] = divB[idx] = 0.0f;
  //               if (is_boundary[2] && pos[1] == grid.guard[1])
  //                 divE[idx] = divB[idx] = 0.0f;
  //               if (is_boundary[3] &&
  //                   pos[1] == grid.dims[1] - grid.guard[1] - 1)
  //                 divE[idx] = divB[idx] = 0.0f;
  //             }
  //           });
  //     },
  //     this->divE, this->divB, this->E, this->B, m_grid_sph.get_grid_ptrs());
  // ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_spherical_1d>::
    compute_b_update_explicit(vector_field<Conf>& B,
                              const vector_field<Conf>& E, double dt) {
  ExecPolicy<Conf>::launch(
      [dt] LAMBDA(auto result, auto e, auto gp) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        // gp is short for grid_ptrs
        ExecPolicy<Conf>::loop(
            Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
              // auto idx = typename Conf::idx_t(n, ext);
              auto pos = get_pos(idx, ext);
              if (grid.is_in_bound(pos)) {
                auto idx_px = idx.inc_x();
                result[1][idx] -= dt * circ1_1d(e, gp.le, idx, idx_px) / gp.Ab[1][idx];
                result[2][idx] -= dt * circ2_1d(e, gp.le, idx, idx_px) / gp.Ab[2][idx];
              }
            });
      },
      B, E, m_grid_sph.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
}


template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_spherical_1d>::
    compute_e_update_explicit(vector_field<Conf>& E,
                              const vector_field<Conf>& B,
                              const vector_field<Conf>& J, double dt) {
    // kernel_launch(
    ExecPolicy<Conf>::launch(
        [dt] LAMBDA(auto result, auto b, auto j, auto gp) {
          auto& grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();
          ExecPolicy<Conf>::loop(
              Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
                // auto pos = idx.get_pos();
                auto pos = get_pos(idx, ext);
                if (grid.is_in_bound(pos)) {
                  auto idx_mx = idx.dec_x();
                  result[0][idx] -= dt * j[0][idx];
                  result[1][idx] += dt * (circ1_1d(b, gp.lb, idx_mx, idx) / gp.Ae[1][idx] - j[1][idx]);
                  result[2][idx] += dt * (circ2_1d(b, gp.lb, idx_mx, idx) / gp.Ae[2][idx] - j[2][idx]);
                }
              });
        },
        E, B, J, m_grid_sph.get_grid_ptrs());
    ExecPolicy<Conf>::sync();
}

}  // namespace Aperture
