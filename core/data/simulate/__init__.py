import parcels

import sys

import numpy as np
from parcels.kernels.advection import AdvectionRK4
from parcels.kernel import Kernel
from parcels.compiler import GNUCompiler
from datetime import timedelta as delta
from datetime import datetime
from parcels.tools.loggers import logger
from parcels import ParticleSet
import time as time_module

from core.data.simulate.trajectories import kernels
from core.data.simulate.trajectories import io


def trajectories(save_path, kernel_path, fieldset, lon, lat, time, ensemble_ids,
                 runtime, output_dt, particle_dt, lon_edge, lat_edge,
                 kernel=parcels.AdvectionRK4, compile_only=False):
    """
    Runs a simulation of particle trajectories using oceanparcels.

    """
    particle_runtime = runtime.total_seconds()
    particle_class = parcels.JITParticle
    
    # oceanparcels accumulates this class level variable between runs, so
    # make sure to reset it here so that particle ids start at 0
    particle_class.lastID = 0

    # a simulation of oceanparcels operates on a particleset
    pset = MyParticleSet(fieldset, particle_class, lon, lat, time=time)

    # define output particlefile to save simulation data to
    output_file = io.ParticleFile(
        name=save_path, particleset=pset, outputdt=output_dt,
        convert_at_end=(not compile_only))

    # we have overriden the output file's export method which requires some
    # extra information
    output_file.ensemble_ids = ensemble_ids
    output_file.times = time
    output_file.runtime = particle_runtime
    output_file.lon_edge = lon_edge
    output_file.lat_edge = lat_edge

    # run the simulation. Particles that throw an error due to going out of
    # bounds are handled by deleting them
    recovery = {parcels.ErrorCode.ErrorOutOfBounds: kernels.delete}
    pset.execute(kernel_path, kernel, runtime=runtime, dt=particle_dt,
                 output_file=output_file, recovery=recovery,
                 verbose_progress=False, compile_only=compile_only)

    # save the simulation output to file
    output_file.close()


# modified from OceanParcels to allow
# (1) reading from a compiled kernel
# (2) exiting after compiling a kernel
class MyParticleSet(parcels.ParticleSet):
    def execute(self, kernel_path, pyfunc=AdvectionRK4, endtime=None,
                runtime=None, dt=1., moviedt=None, recovery=None,
                output_file=None, movie_background_field=None,
                verbose_progress=None, postIterationCallbacks=None,
                callbackdt=None, compile_only=False):
        # check if pyfunc has changed since last compile. If so, recompile
        if self.kernel is None or (self.kernel.pyfunc is not pyfunc and self.kernel is not pyfunc):
            # Generate and store Kernel
            if isinstance(pyfunc, Kernel):
                self.kernel = pyfunc
            else:
                self.kernel = self.Kernel(pyfunc)
            # Prepare JIT kernel execution
            if self.ptype.uses_jit:
                if not kernel_path.exists():
                    self.kernel.remove_lib()
                    # MODIFICATION (1)
                    self.kernel.lib_file = str(kernel_path.with_suffix('.so'))

                    cppargs = ['-DDOUBLE_COORD_VARIABLES'] if self.lonlatdepth_dtype == np.float64 else None
                    self.kernel.compile(compiler=GNUCompiler(cppargs=cppargs))
                else:
                    self.kernel.lib_file = str(kernel_path.with_suffix('.so'))
                # MODIFICATION (2)
                if compile_only:
                    sys.exit()
                self.kernel.load_lib()
                    
        # Convert all time variables to seconds
        if isinstance(endtime, delta):
            raise RuntimeError('endtime must be either a datetime or a double')
        if isinstance(endtime, datetime):
            endtime = np.datetime64(endtime)
        if isinstance(endtime, np.datetime64):
            if self.time_origin.calendar is None:
                raise NotImplementedError('If fieldset.time_origin is not a date, execution endtime must be a double')
            endtime = self.time_origin.reltime(endtime)
        if isinstance(runtime, delta):
            runtime = runtime.total_seconds()
        if isinstance(dt, delta):
            dt = dt.total_seconds()
        outputdt = output_file.outputdt if output_file else np.infty
        if isinstance(outputdt, delta):
            outputdt = outputdt.total_seconds()
        if isinstance(moviedt, delta):
            moviedt = moviedt.total_seconds()
        if isinstance(callbackdt, delta):
            callbackdt = callbackdt.total_seconds()

        assert runtime is None or runtime >= 0, 'runtime must be positive'
        assert outputdt is None or outputdt >= 0, 'outputdt must be positive'
        assert moviedt is None or moviedt >= 0, 'moviedt must be positive'

        mintime, maxtime = self.fieldset.gridset.dimrange('time_full')
        if np.any(np.isnan(self.particle_data['time'])):
            self.particle_data['time'][np.isnan(self.particle_data['time'])] = mintime if dt >= 0 else maxtime

        # Derive _starttime and endtime from arguments or fieldset defaults
        if runtime is not None and endtime is not None:
            raise RuntimeError('Only one of (endtime, runtime) can be specified')
        _starttime = self.particle_data['time'].min() if dt >= 0 else self.particle_data['time'].max()
        if self.repeatdt is not None and self.repeat_starttime is None:
            self.repeat_starttime = _starttime
        if runtime is not None:
            endtime = _starttime + runtime * np.sign(dt)
        elif endtime is None:
            mintime, maxtime = self.fieldset.gridset.dimrange('time_full')
            endtime = maxtime if dt >= 0 else mintime

        execute_once = False
        if abs(endtime-_starttime) < 1e-5 or dt == 0 or runtime == 0:
            dt = 0
            runtime = 0
            endtime = _starttime
            logger.warning_once("dt or runtime are zero, or endtime is equal to Particle.time. "
                                "The kernels will be executed once, without incrementing time")
            execute_once = True

        self.particle_data['dt'][:] = dt

        # First write output_file, because particles could have been added
        if output_file:
            output_file.write(self, _starttime)
        if moviedt:
            self.show(field=movie_background_field, show_time=_starttime, animation=True)

        if moviedt is None:
            moviedt = np.infty
        if callbackdt is None:
            interupt_dts = [np.infty, moviedt, outputdt]
            if self.repeatdt is not None:
                interupt_dts.append(self.repeatdt)
            callbackdt = np.min(np.array(interupt_dts))
        time = _starttime
        if self.repeatdt:
            next_prelease = self.repeat_starttime + (abs(time - self.repeat_starttime) // self.repeatdt + 1) * self.repeatdt * np.sign(dt)
        else:
            next_prelease = np.infty if dt > 0 else - np.infty
        next_output = time + outputdt if dt > 0 else time - outputdt
        next_movie = time + moviedt if dt > 0 else time - moviedt
        next_callback = time + callbackdt if dt > 0 else time - callbackdt
        next_input = self.fieldset.computeTimeChunk(time, np.sign(dt))

        tol = 1e-12
        if verbose_progress is None:
            walltime_start = time_module.time()
        if verbose_progress:
            pbar = self.__create_progressbar(_starttime, endtime)
        while (time < endtime and dt > 0) or (time > endtime and dt < 0) or dt == 0:
            if verbose_progress is None and time_module.time() - walltime_start > 10:
                # Showing progressbar if runtime > 10 seconds
                if output_file:
                    logger.info('Temporary output files are stored in %s.' % output_file.tempwritedir_base)
                    logger.info('You can use "parcels_convert_npydir_to_netcdf %s" to convert these '
                                'to a NetCDF file during the run.' % output_file.tempwritedir_base)
                pbar = self.__create_progressbar(_starttime, endtime)
                verbose_progress = True
            if dt > 0:
                time = min(next_prelease, next_input, next_output, next_movie, next_callback, endtime)
            else:
                time = max(next_prelease, next_input, next_output, next_movie, next_callback, endtime)
            self.kernel.execute(self, endtime=time, dt=dt, recovery=recovery, output_file=output_file,
                                execute_once=execute_once)
            if abs(time-next_prelease) < tol:
                pset_new = ParticleSet(fieldset=self.fieldset, time=time, lon=self.repeatlon,
                                       lat=self.repeatlat, depth=self.repeatdepth,
                                       pclass=self.repeatpclass, lonlatdepth_dtype=self.lonlatdepth_dtype,
                                       partitions=False, pid_orig=self.repeatpid, **self.repeatkwargs)
                p = pset_new.data_accessor()
                for i in range(pset_new.size):
                    p.set_index(i)
                    p.dt = dt
                self.add(pset_new)
                next_prelease += self.repeatdt * np.sign(dt)
            if abs(time-next_output) < tol:
                if output_file:
                    output_file.write(self, time)
                next_output += outputdt * np.sign(dt)
            if abs(time-next_movie) < tol:
                self.show(field=movie_background_field, show_time=time, animation=True)
                next_movie += moviedt * np.sign(dt)
            # ==== insert post-process here to also allow for memory clean-up via external func ==== #
            if abs(time-next_callback) < tol:
                if postIterationCallbacks is not None:
                    for extFunc in postIterationCallbacks:
                        extFunc()
                next_callback += callbackdt * np.sign(dt)
            if time != endtime:
                next_input = self.fieldset.computeTimeChunk(time, dt)
            if dt == 0:
                break
            if verbose_progress:
                pbar.update(abs(time - _starttime))

        if output_file:
            output_file.write(self, time)
        if verbose_progress:
            pbar.finish()