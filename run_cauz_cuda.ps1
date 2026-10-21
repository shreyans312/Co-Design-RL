$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$experiments = @(
    @{ Name = "improved_training_seed43"; Extra = @("--model", "ddpg_lognormal_improved", "--design-warmup-fraction", "0.10", "--exploration-noise-std", "0.30", "--final-exploration-noise-std", "0.05") }
    @{ Name = "original_std3_seed43"; Extra = @("--model", "ddpg_lognormal") }
    @{ Name = "original_std08_seed43"; Extra = @("--model", "ddpg_lognormal", "--design-initial-std", "0.8") }
)

foreach ($experiment in $experiments) {
    $output = Join-Path (Join-Path $PSScriptRoot "runs") $experiment.Name
    if (Test-Path $output) {
        throw "Output already exists: $output. Move or rename it before starting a new run"
    }
}

if ($env:CONDA_PREFIX) {
    $python = Join-Path $env:CONDA_PREFIX "python.exe"
    if (-not (Test-Path $python)) { throw "Active Conda environment has no python.exe: $env:CONDA_PREFIX" }
} else {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Activate a Python 3.12 Conda environment or install the Windows 'py' launcher"
    }
    $python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) {
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Could not create the Python 3.12 environment" }
    }
}

& $python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"; print(sys.executable)'
if ($LASTEXITCODE -ne 0) { throw "Python version check failed; no training was started" }

& $python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }

& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Requirements installation failed" }

& $python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
if ($LASTEXITCODE -ne 0) { throw "CUDA check failed; no training was started" }

& $python -m unittest test_rl.RLConfigTests.test_ddpg_cuda_actor_and_critic_update
if ($LASTEXITCODE -ne 0) { throw "CUDA actor/critic test failed; no training was started" }

$common = @(
    "-u", "train_rl.py",
    "--dataset", "cauz",
    "--device", "cuda",
    "--seed", "43",
    "--data-seed", "42",
    "--iterations", "500",
    "--episodes-per-iteration", "32",
    "--design-scoring", "training",
    "--matched-validation",
    "--validation-episodes", "32",
    "--validation-every", "5",
    "--save-every", "5",
    "--log-every", "5",
    "--torch-threads", "1"
)

foreach ($experiment in $experiments) {
    $relativeOutput = Join-Path "runs" $experiment.Name
    $output = Join-Path $PSScriptRoot $relativeOutput
    New-Item -ItemType Directory -Path $output | Out-Null
    $runArguments = $common + $experiment.Extra + @("--output-directory", $relativeOutput)
    $process = Start-Process -FilePath $python -ArgumentList $runArguments -WorkingDirectory $PSScriptRoot -RedirectStandardOutput (Join-Path $output "train.log") -RedirectStandardError (Join-Path $output "error.log") -PassThru
    Write-Host "$($experiment.Name) started as PID $($process.Id)"
    Write-Host "  Get-Content '$output\train.log' -Tail 30 -Wait"
    Write-Host "  Get-Content '$output\error.log' -Tail 30 -Wait"
}
