# Deploy the review app on Oracle Cloud Always Free

A free, always-on copy of the app on a VM you own. Oracle's Always Free tier includes an Arm (Ampere A1) VM
with up to 2 OCPUs and 12 GB of memory, which fits the app (Llama 3.2 1B in float32 + LoRA + bge-small needs
about 6 GB). It does not sleep, so no keep-awake ping is needed. Time: about 1 hour, most of it waiting for builds.

You need:
- A Hugging Face read token from your own account (`pranatipradhan123`). The account has accepted the Llama 3.2
  licence and is an admin of the `hdfc-capstone` org, which holds the adapters and FAQ index, so nothing needs copying.
- A credit or debit card for Oracle's identity check (Always Free resources are not charged).

## 1. Create the VM (Oracle website)

1. Sign up at https://signup.cloud.oracle.com. Pick a **home region** close to you (e.g. India West (Mumbai)
   or India South (Hyderabad)); Always Free VMs can only be created there and it can't be changed later.
2. **Compute → Instances → Create instance**.
   - Image: **Canonical Ubuntu 24.04**.
   - Shape: **Ampere → VM.Standard.A1.Flex**, **2 OCPUs, 12 GB memory** (check it says "Always Free-eligible").
   - Networking: keep "Assign a public IPv4 address" on.
   - SSH keys: **Generate a key pair for me** and download the private key (e.g. `oracle.key`).
   - Boot volume: default (about 50 GB) is enough.
3. If it says **Out of capacity**, try another availability domain or try again later (common for free Arm VMs).
4. Note the **Public IP address** on the instance page.
5. Open the web ports: instance page → **Subnet** → **Default security list** → **Add ingress rules**:
   source `0.0.0.0/0`, TCP, destination ports `80,443`.

## 2. Install Docker on the VM (your Mac's Terminal)

```bash
chmod 600 ~/Downloads/oracle.key
ssh -i ~/Downloads/oracle.key ubuntu@<PUBLIC_IP>
```

Then on the VM:

```bash
# Ubuntu on Oracle blocks web ports in its own firewall too; open 80 and 443
sudo iptables -I INPUT 6 -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
exit
```

Log in again with the same `ssh` command so the `docker` group applies.

## 3. Get the code and settings

```bash
git clone https://github.com/pranatipradhanwork-alt/Hdfc-bank-custom-llm-pipeline-project.git hdfc
cd hdfc
cp .env.example .env
nano .env
```

In `.env` set (save with Ctrl+O, Enter, Ctrl+X):
- `HF_TOKEN=` your Hugging Face read token (https://huggingface.co/settings/tokens).
- `APP_KEY=` any long random string (`openssl rand -hex 24` prints one).
- `METRICS_TOKEN=` another random string.
- `SITE_ADDRESS=` the public IP with dashes plus `.sslip.io`, e.g. IP `140.238.1.2` → `140-238-1-2.sslip.io`
  (a free hostname that points at your IP, so Caddy can get an HTTPS certificate).

## 4. Start it

```bash
docker compose -f deploy/docker-compose.vm.yml up -d --build
docker compose -f deploy/docker-compose.vm.yml logs -f app
```

The first build takes about 15–20 minutes; the first start then downloads the base model (~2.5 GB) and checks the
adapter checksum against `registry.json`. When the log shows `Uvicorn running`, press Ctrl+C (the app keeps
running) and open `https://<SITE_ADDRESS>`. The sign-in accounts are the same as on the Space.

Check: `curl https://<SITE_ADDRESS>/v1/health` shows the live model `llama_v3_1ep`.

## Update to the latest `main`

```bash
cd ~/hdfc && git pull && docker compose -f deploy/docker-compose.vm.yml up -d --build
```

## Good to know

- Answers run on 2 Arm cores, so expect a few seconds per answer (similar to the CPU Space).
- Oracle may reclaim an Always Free VM that is idle for 7 days (CPU, network **and** memory all under 20%).
  The loaded model keeps memory use around 50%, so this should not apply while the app runs.
- Downloaded models are kept in Docker volumes. Data written at runtime (audit log, new team assistants) survives
  a VM reboot but is reset when you update with `--build`, as on the Space.
- Stop it: `docker compose -f deploy/docker-compose.vm.yml down` (add `-v` to also delete the downloaded models).
