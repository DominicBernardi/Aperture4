/*
 * Copyright (c) 2021 Alex Chen.
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

#include "core/cuda_control.h"
#include "core/math.hpp"
// #include "core/multi_array_exp.hpp"
#include "core/particles.h"
#include "data/data_array.hpp"
#include "data/fields.h"
#include "framework/environment.h"
#include "grid_sph_1d.hpp"
#include "systems/helpers/filter_field.hpp"
#include "systems/helpers/ptc_update_helper.hpp"
#include "systems/physics/pushers.hpp"
#include "systems/physics/pusher_simple.hpp"

namespace Aperture {

template <typename Conf, template <class> class Pusher = pusher_simple>
class coord_policy_spherical_1d_base {
 public:
  using value_t = typename Conf::value_t;
  using grid_type = grid_sph_1d_t<Conf>;

  coord_policy_spherical_1d_base(const grid_t<Conf>& grid) :
      m_grid(dynamic_cast<const grid_type&>(grid)),
      m_pusher(grid) {}
  ~coord_policy_spherical_1d_base() = default;

  // Static coordinate functions
  HD_INLINE static value_t weight_func(value_t x1, value_t x2,
                                       value_t x3 = 0.0f) {
    return 1.0;
  }

  HD_INLINE static value_t x1(value_t x) { return grid_type::radius(x); }
  HD_INLINE static value_t x2(value_t x) { return grid_type::theta(x); }
  HD_INLINE static value_t x3(value_t x) { return x; }

  void init() {
    m_pusher.init();
  }

  // Inline functions to be called in the particle update loop
  template <typename PtcContext>
  HD_INLINE void update_ptc(const Grid<Conf::dim, value_t>& grid,
                            const extent_t<Conf::dim>& ext, PtcContext& context,
                            index_t<Conf::dim>& pos, value_t dt) const {
    if (!check_flag(context.flag, PtcFlag::ignore_EM)) {
      m_pusher.push(grid, ext, context, pos, dt);
    }
    move_ptc(grid, context, pos, dt);
  }

  // Abstracted moving routine that is shared by both ptc and ph
  template <typename PtcContext>
  HD_INLINE void move_ptc(const Grid<Conf::dim, value_t>& grid,
                          PtcContext& context, index_t<Conf::dim>& pos,
                          value_t dt) const {
    context.x[1] = M_PI/2;
    context.x[2] = 0.0;
    // Global position in sph coord
    vec_t<value_t, 3> x_global_old(
        grid.template coord<0>(pos[0], context.x[0]), context.x[1], context.x[2]);
    // Global position in cartesian coord
    vec_t<value_t, 3> x_global_cart(grid_type::radius(x_global_old[0]), 0.0, 0.0);

    // Transform momentum vector to cartesian
    value_t pr = context.p[0], pth = context.p[1], pph = context.p[2];
    context.p[0] = pr;     // p_x
    context.p[1] = pph;    // p_y
    context.p[2] = -pth;   // p_z

    // Move in Cartesian coordinates
    x_global_cart += context.p * (dt / context.gamma);

    // Compute the new spherical location
    vec_t<value_t, 3> x_global_sph_new =
        grid_type::coord_from_cart(x_global_cart);

    // Transform the momentum vector to spherical at the new location
    grid_type::vec_from_cart(context.p, x_global_sph_new);
#pragma unroll
    for (int i = 0; i < Conf::dim; i++) {
      context.new_x[i] = context.x[i] + (x_global_sph_new[i] - x_global_old[i]) * grid.inv_delta[i];
      context.dc[i] = std::floor(context.new_x[i]);
      pos[i] += context.dc[i];
      context.new_x[i] -= (value_t)context.dc[i];
    }
#pragma unroll
    for (int i = Conf::dim; i < 3; i++) {
      context.new_x[i] = x_global_sph_new[i];
    }
  }

  // Inline functions to be called in the photon update loop
  HD_INLINE void update_ph(const Grid<Conf::dim, value_t>& grid,
                           ph_context<Conf::dim, value_t>& context,
                           index_t<Conf::dim>& pos, value_t dt) const {
    move_ptc(grid, context, pos, dt);
  }

  // Extra processing routines
  template <typename ExecPolicy>
  void process_J_Rho(vector_field<Conf>& J, data_array<scalar_field<Conf>>& Rho,
                     scalar_field<Conf>& rho_total,
                     value_t dt, bool process_rho) const {
    auto num_species = Rho.size();
    ExecPolicy::launch(
        [dt, num_species, process_rho] LAMBDA(auto j, auto rho, auto rho_total,
                                              auto grid_ptrs) {
          auto& grid = ExecPolicy::grid();
          auto ext = grid.extent();
          // grid.cell_size() is simply the product of all deltas
          // auto w = grid.cell_size() / dt;
          auto w = grid.cell_size();
          ExecPolicy::loop(
              Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
                j[0][idx] *= w / grid_ptrs.Ae[0][idx];
                j[1][idx] *= w / grid_ptrs.Ae[1][idx];
                j[2][idx] *= w / grid_ptrs.Ae[2][idx];

                rho_total[idx] /= grid_ptrs.dV[idx];
                for (int n = 0; n < num_species; n++) {
                  rho[n][idx] /= grid_ptrs.dV[idx];
                }
              });
        },
        J, Rho, rho_total, m_grid.get_grid_ptrs());
    ExecPolicy::sync();
  }

  template <typename ExecPolicy>
  void filter_field(vector_field<Conf>& field,
                    typename Conf::multi_array_t& tmp,
                    const vec_t<bool, Conf::dim * 2>& is_boundary,
                    int extend = 0) const {
    filter_field_component<ExecPolicy>(field.at(0), tmp, m_grid.m_Ae[0],
                                       is_boundary, field.stagger(0), extend);
    filter_field_component<ExecPolicy>(field.at(1), tmp, m_grid.m_Ae[1],
                                       is_boundary, field.stagger(1), extend);
    filter_field_component<ExecPolicy>(field.at(2), tmp, m_grid.m_Ae[2],
                                       is_boundary, field.stagger(2), extend);
  }

  template <typename ExecPolicy>
  void filter_field(scalar_field<Conf>& field,
                    typename Conf::multi_array_t& tmp,
                    const vec_t<bool, Conf::dim * 2>& is_boundary,
                    int extend = 0) const {
    filter_field_component<ExecPolicy>(field.at(0), tmp, m_grid.m_dV,
                                       is_boundary, field.stagger(), extend);
  }

 private:
  const grid_type& m_grid;
  Pusher<Conf> m_pusher;
};

template <typename Conf>
using coord_policy_spherical_1d = coord_policy_spherical_1d_base<Conf, pusher_simple>;

}  // namespace Aperture
