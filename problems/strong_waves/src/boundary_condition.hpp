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

#pragma once

#include "core/math.hpp"
#include "data/fields.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/domain_comm.h"
#include "systems/grid.h"
#include "utils/nonown_ptr.hpp"
#include "utils/util_functions.h"
#include <memory>

namespace Aperture {

// Launches a wave travelling in +x from the lower x boundary of a 1D box. The
// wave vector is along x, and the background field B0 lies in the x-y plane.
// The wave electric field is along z, i.e. perpendicular to the k-B0 plane,
// which is the fast magnetosonic polarization. With B0 = 0 this is simply a
// linearly polarized vacuum electromagnetic wave. The wave magnetic field is
// delta B_y = -E_z, so that E x B points in +x.
template <typename Conf, template <class> class ExecPolicy>
class boundary_condition : public system_t {
 protected:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf, ExecPolicy>* m_comm = nullptr;

  double m_A0 = 1.0;       // wave amplitude, E_z = -delta B_y = A0 sin(phase)
  double m_freq = 5.0;     // wave frequency (not angular frequency)
  int m_num_lambda = 4;    // number of wavelengths in the wave train

  nonown_ptr<vector_field<Conf>> E, B;

 public:
  static std::string name() { return "boundary_condition"; }

  boundary_condition(const grid_t<Conf>& grid,
                     const domain_comm<Conf, ExecPolicy>* comm = nullptr)
      : m_grid(grid), m_comm(comm) {}

  void init() override {
    sim_env().get_data("Edelta", E);
    sim_env().get_data("Bdelta", B);

    sim_env().params().get_value("A0", m_A0);
    sim_env().params().get_value("wave_freq", m_freq);
    sim_env().params().get_value("num_lambda", m_num_lambda);
  }

  void update(double dt, uint32_t step) override {
    if (m_comm != nullptr && !m_comm->domain_info().is_boundary[0]) return;

    typedef typename Conf::idx_t idx_t;
    typedef typename Conf::value_t value_t;

    // This system runs after the field solver, so E is at t + dt and B is at
    // t + dt/2, where t is the time at the start of this step
    value_t time_e = sim_env().get_time() + dt;
    value_t time_b = sim_env().get_time() + 0.5 * dt;
    value_t A0 = m_A0;
    value_t omega = 2.0 * M_PI * m_freq;
    value_t phase_max = 2.0 * M_PI * m_num_lambda;
    value_t x0 = m_grid.lower[0];
    int st_e = E->stagger(2)[0];
    int st_b = B->stagger(1)[0];

    // Waveform of a right-moving wave train that enters the box at x0 at t = 0
    auto wave = [A0, omega, phase_max, x0] LAMBDA(value_t t, value_t x) {
      value_t phase = omega * (t - (x - x0));
      if (phase > 0.0 && phase < phase_max)
        return A0 * math::sin(phase);
      else
        return value_t(0.0);
    };

    ExecPolicy<Conf>::launch(
        [=] LAMBDA(auto e, auto b) {
          auto& grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();
          // E_z sits on the cell edge, so n0 = guard is exactly on the
          // boundary and is driven as well. B_y sits at cell centers, so only
          // the guard cells are driven.
          ExecPolicy<Conf>::loop(0, grid.guard[0] + 1, [&] LAMBDA(auto n0) {
            auto idx = idx_t(index_t<1>(n0), ext);
            e[1][idx] = 0.0;
            e[2][idx] = wave(time_e, grid.coord(0, n0, st_e));
            if (n0 < grid.guard[0]) {
              b[1][idx] = -wave(time_b, grid.coord(0, n0, st_b));
              b[2][idx] = 0.0;
            }
          });
        },
        E, B);
    ExecPolicy<Conf>::sync();
  }
};

}  // namespace Aperture
