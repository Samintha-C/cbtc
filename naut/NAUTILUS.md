# Nautilus procedures

How work runs on the NRP Nautilus cluster for this project, and what Claude is
and isn't allowed to do there. The same file lives in each project's `naut/`
folder (`cbtc`, `cbvlam`); keep them in sync.

## The cluster

- **Namespace:** `wenglab-interpretable-ai`. It's shared lab quota, so anything
  left running costs everyone.
- **Storage:** the shared volume `sc-rwx-vol`, mounted at `/sc-rwx-vol`. Each
  project keeps its data, caches and results under its own folder there (e.g.
  `/sc-rwx-vol/fedmedcbm/`). Some CB-VLAM jobs also use `sc-cbint-vol`,
  mounted at `/sc-cbint-vol`.
- **Results live on the volume, not in `kubectl`.** Finished jobs are cleaned
  up after 24 h (`ttlSecondsAfterFinished: 86400`), so after a day
  `kubectl get jobs` shows nothing. Check the volume before assuming a job never
  ran.

## How jobs are written and run

1. **Job files live in `naut/`.** For sweeps, a small Python generator
   (`naut/gen_*.py`) writes one YAML per configuration, so the settings come from
   code rather than hand edits.
2. **A pod sets itself up from GitHub.** Each job clones the project repo (plus
   any sibling repos it needs) into `$WORKSPACE`, runs `pip install -r
   requirements.txt` on top of a stock image, then runs the entry script. The
   repo must be reachable from the pod: a private repo needs a token stored as a
   Kubernetes secret.
3. **Outputs go to the shared volume**, under the project folder, one
   directory per run (with an `args.txt` recording the exact flags).
4. **Check the file before submitting:**
   `kubectl apply --dry-run=server -f <file>` validates against the cluster
   without creating anything.
5. **Samintha submits.** See the rules below.

### Cluster rules that jobs must satisfy

- **Requests must equal limits** (or be within 20%) for CPU, memory and GPU.
- **GPU utilization must stay above about 40%.** A GPU sitting idle is a policy
  violation. Work that barely uses the GPU (e.g. a tiny final layer) should run
  as a CPU job instead; the Fed-CBM pipeline splits phases into CPU and GPU jobs
  for this reason.
- **Typical GPU job:** 12Gi RAM, 4 CPU, 1 GPU, plus a `/dev/shm` memory volume
  (6Gi) for the DataLoader. `/dev/shm` must fit workers × prefetch × batch.
- **Watch DataLoader workers on large datasets.** Each forked worker can inherit
  gigabytes of in-memory data; this OOM-killed Places365/ImageNet runs. Code that
  forces `num_workers=0` when data is already in memory does so on purpose.
- **Time estimates:** the first run on a dataset pays for feature extraction
  (hours to a day on ImageNet-scale data); later runs that hit the cache take a
  fraction of that. Don't extrapolate remaining time from cold runs.

## Watching jobs (read-only, always fine)

```bash
kubectl get jobs,pods -n wenglab-interpretable-ai
kubectl logs <pod> -n wenglab-interpretable-ai --timestamps | tail -50
kubectl describe pod <pod> -n wenglab-interpretable-ai   # scheduling / OOM reasons
```

## The copy pod: reading the shared volume

`sc-rwx-copy-pod` is a tiny Ubuntu pod (100m CPU, 256Mi, no GPU) that mounts
`sc-rwx-vol` so results can be inspected and copied off the cluster. Its
manifest is `fed_lfc_cbm/naut/copy.yaml` (in the TrustworthyMLLab workspace).

It runs `sleep 21600`, so it **exits on its own after 6 hours** and then shows
as `Completed`/`Failed`. That's normal; restart it.

```bash
# start (or restart after it has exited)
kubectl delete pod sc-rwx-copy-pod -n wenglab-interpretable-ai --ignore-not-found
kubectl apply -f ../fed_lfc_cbm/naut/copy.yaml
#the above assumes that you are currently in a project directory within the greater TrustworthyMLlab dir, since fed_lfc_cbm is another project i have that just happens to be where i started using the copy pod for miscallaneous taks
kubectl wait --for=condition=Ready pod/sc-rwx-copy-pod -n wenglab-interpretable-ai --timeout=120s

# look around / read a file
kubectl exec sc-rwx-copy-pod -n wenglab-interpretable-ai -- ls /sc-rwx-vol/<project>/
kubectl exec sc-rwx-copy-pod -n wenglab-interpretable-ai -- cat /sc-rwx-vol/<project>/<run>/args.txt

# copy results down
kubectl cp wenglab-interpretable-ai/sc-rwx-copy-pod:/sc-rwx-vol/<project>/<path> ./<local-path>
```

The manifest does **not** mount the volume read-only, so `exec` can write to
it. Only use it to read and copy.

## What Claude may do

- **Allowed without asking:** read-only commands (`kubectl get`, `describe`,
  `logs`, `apply --dry-run=server`), and the **copy pod only**: create it,
  delete it when it is stale or has exited, and `exec`/`cp` from it to read
  results.
- **Never, unless explicitly asked:** submit, apply, create or delete anything
  else. That includes jobs, other pods, "quick" lister pods, secrets and
  volumes. Claude writes the job YAML and validates it with a dry run;
  **Samintha submits it.**
- **Why:** GPU jobs spend shared lab quota, and a stray pod is Samintha's
  problem to clean up. The copy pod is the one exception because it dies every
  6 hours and holds no GPU, so restarting it through a human was pure friction.

## Credentials

- Never commit tokens (Hugging Face, GitHub, Kaggle, API keys). Keep any local
  token file out of git via `.gitignore` (e.g. `naut/tokens.txt`).
- Jobs get credentials from Kubernetes secrets created by Samintha, e.g.
  `kubectl create secret generic <name> --from-literal=token=... -n wenglab-interpretable-ai`,
  read in the YAML with `secretKeyRef`. Never paste a token into a YAML.
