"""Extract one qualified archive member without modifying the installed archive.

Two members have the name xerbla.c.o. Only the pinned second member defines
the original weak xerbla_ / strong __xerbla pair. Archive and member identity
are checked before writing; unknown providers require a new qualification.
"""
from pathlib import Path
import hashlib
import sys

archive, output = map(Path, sys.argv[1:])
data = archive.read_bytes()
if hashlib.sha256(data).hexdigest() != "609d9d208092fcf6af3c67c86ca561dcdce19ec16fae6afd2a0577b940c46413":
    raise SystemExit("provider archive identity mismatch")
header = data[3988260:3988320]
if header[:16].decode().strip() != "xerbla.c.o/" or int(header[48:58]) != 1664 or header[58:] != b"`\n":
    raise SystemExit("provider member metadata mismatch")
member = data[3988320:3989984]
if hashlib.sha256(member).hexdigest() != "237baff03987eb5fbf97cadfd3316467ef87dcd34924aff2d0d8981bbb45a806":
    raise SystemExit("provider handler identity mismatch")
output.parent.mkdir(parents=True, exist_ok=True)
if not output.exists() or output.read_bytes() != member:
    output.write_bytes(member)
