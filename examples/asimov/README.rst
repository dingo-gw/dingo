Asimov examples and integration smoke test
=========================================

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
these four files and saves its overridden copies as artifacts. It installs
Asimov >=0.8,<0.9 and bilby_pipe >=1.10,<1.11 with the current Dingo checkout.
It runs on pull requests to ``main``, ``asimov-v0.7``, or ``asimov-IR1``,
pushes to ``main``, and manual dispatch.

The test exercises real Asimov configuration, Dingo argument parsing and DAG
construction, HTCondor scheduling and file transfers, and PESummary export
and page generation. The Dingo generation, sampling, and importance-sampling
executables are replaced by test workers in ``tests/e2e/mock_dingo.py``.
The DAG builder's model loader is also replaced with a metadata-only loader.
The workers write zero strain, 32 deterministic synthetic samples, unit
weights, and fixed evidence. No inference, likelihood evaluation, GNPE
iterations, or density recovery runs. Frequency assertions check configuration
propagation into result metadata, not likelihood ASD masking or scientific
accuracy.

``tests/e2e/model_fixtures.py`` writes metadata-only checkpoints without
network weights and a flat PSD. The multibanded domain has nodes at 20, 64,
and 128 Hz, with 0.25 Hz spacing in the first band and 0.5 Hz in the second.
The event uses the uniform 0.25 Hz base grid. Each analysis supplies
``available networks`` with an incompatible candidate first, followed by a
compatible transformer or GNPE candidate. This exercises automatic frequency
compatibility filtering and checks that transformer selection clears the
initializer while GNPE retains it. It does not test ranking several compatible
networks or the explicit ``networks`` override.

Both analyses run ``asimov apply -> manage build -> manage submit -> monitor``.
HTCondor runs the mock Dingo workers with one CPU and 2 GB requested memory
per job. The isolated mini pool has one CPU and 8 GB advertised memory, so
jobs run serially. The larger memory advertisement accommodates
asimov-pesummary's fixed 8192 MB request. Each analysis has a separate
PESummary production with a ``needs:`` dependency. After monitoring Dingo's
completion, a second build/submit step starts those productions. PESummary
uses one process and ten skymap samples per analysis.

Assertions in ``assert_jobs_succeeded()`` and ``assert_outputs()`` in
``tests/e2e/asimov_htcondor.py`` check job exits, selected checkpoint paths,
initializer selection, final ``finished`` ledger states, and readable
multibanded results with 32 samples and finite synthetic evidence. They check
the H1 band 24--96 Hz and L1 band 28--112 Hz in result metadata. Each
PESummary run must produce a nonempty ``home.html`` and readable posterior
samples under the expected analysis label.

The job has a 30-minute overall timeout, a 10-minute wait per Dingo DAG,
and a 15-minute wait for PESummary. The driver uses ``condor_wait`` and sends
command output directly to the terminal. It writes no diagnostic snapshots
or success-report files. Blueprints, metadata fixtures, mock executables,
native job logs, results and summary pages are retained for five days in CI.
This dedicated script is not collected by the normal pytest suite.

To run in an equivalent, already configured isolated pool, as an
unprivileged user with the dependencies installed::

    python -m pip install -e . 'asimov>=0.8,<0.9' 'bilby_pipe>=1.10,<1.11' asimov-pesummary
    python tests/e2e/asimov_htcondor.py /path/to/empty/e2e_project

The pool must advertise at least 8192 MB of memory for PESummary. Each run
needs an empty output directory; old scheduler history does not affect it.
The driver stores its Git identity in the disposable project, leaving your
normal Git configuration untouched. Success prints
``Passed: network selection, job orchestration, and PESummary.`` and exits zero.
