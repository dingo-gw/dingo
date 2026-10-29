Asimov examples and CI
=====================

Apply ``project_dingo.yaml``, then ``event_dingo.yaml``, then
``analysis_dingo.yaml`` and ``analysis_pesummary.yaml`` with
``-e GW150914_095045``. Replace the checkpoint
paths and accounting settings first. For a transformer, set ``networks.model
init`` to null; GNPE requires both checkpoints. Frequency bounds belong in
``likelihood`` and are keyed by interferometer. Event time and interferometers
are top-level event fields. Install ``asimov-pesummary`` for post-processing.
The PESummary analysis uses ``needs: [Exp0]`` to wait for Dingo; update this
dependency if you rename the Dingo analysis. Match its waveform approximant
and reference frequency to the network. PESummary defaults belong under
``postprocessing.pesummary`` in the project blueprint.

The GitHub Actions workflow in ``.github/workflows/asimov-e2e.yml`` loads
these four files directly and saves its overridden copies as artifacts.
It uses Asimov 0.8.0 and bilby_pipe 1.10.2, with the current Dingo checkout.
It runs on pull requests to main, pushes to main, and manual dispatch.

The test replaces production resource settings and paths with one CPU and
2 GB requested memory per Dingo job, tiny initialized CPU checkpoints, one four-second
Gaussian-noise segment, and 32 posterior samples. There is no waveform
training dataset, SVD computation, or training of the inference checkpoints.
The checkpoints contain fixed prior/standardization metadata and random
weights, sufficient to exercise loading and inference through Asimov.
All checkpoints use a multibanded frequency domain with nodes at 20, 64,
and 128 Hz: 0.25 Hz spacing in the first band and 0.5 Hz in the second.
The PSD and generated strain use the uniform 0.25 Hz base grid, also used
for the importance-sampling likelihood.
GNPE uses two iterations and fits its density recovery flow for one epoch
on 64 proxy samples. This fit is part of GNPE inference, needed to obtain
proposal densities for importance sampling. The isolated HTCondor mini pool
has one CPU and 8 GB advertised memory; jobs run serially. The larger memory
advertisement accommodates asimov-pesummary's fixed 8192 MB request.
PESummary uses one process and ten skymap samples per analysis, with spin
evolution and explicit posterior regeneration disabled. No GPU, external strain,
or production checkpoints are needed. These small models
test execution and configuration propagation, not scientific accuracy.

Both architectures run ``asimov apply -> manage build -> manage submit ->
monitor``, with real HTCondor generation, sampling and importance-sampling
jobs. Each analysis has a separate PESummary production with a ``needs:``
dependency. After monitoring Dingo's completion, a second build/submit step
starts those productions through Asimov.

All assertions are together in ``assert_outputs()`` near the top of
``tests/e2e/asimov_htcondor.py``. They check successful job exits, final
``finished`` ledger states, multibanded weighted results with 32 samples and
finite evidence, and the H1 band 24--96 Hz and L1 band 28--112 Hz in the
results and likelihood ASD masks. For each PESummary run, they require a
nonempty ``home.html`` and readable posterior samples under the expected
analysis label. Checkpoint construction is separate in ``model_fixtures.py``.

The job has a 30-minute overall timeout, a 10-minute wait per Dingo DAG,
and a 15-minute wait for PESummary. The driver uses ``condor_wait`` and sends
command output directly to the terminal. It writes no diagnostic snapshots
or success-report files. Blueprints, native job logs, results and summary
pages are retained for five days in CI; fixture checkpoints are not uploaded.
This dedicated script is not collected by the normal pytest suite.

To run in an equivalent, already configured isolated pool, as an
unprivileged user with the dependencies installed::

    python -m pip install -e . 'asimov==0.8.0' 'bilby_pipe==1.10.2' asimov-pesummary
    python tests/e2e/asimov_htcondor.py /path/to/empty/e2e_project

The pool must advertise at least 8192 MB of memory for PESummary. Each run
needs an empty output directory; old scheduler history does not affect it.
The driver stores its Git identity in the disposable project, leaving your
normal Git configuration untouched. Success prints
``Passed: transformer, GNPE, frequency bounds, and PESummary.`` and exits zero.
