def xdb():
    import atexit;
    import os;
    import pdb;
    import pty;
    import subprocess;

    master, slave = pty.openpty()
    xterm = subprocess.Popen(['/usr/bin/xterm', '-Stt%d' % master])
    atexit.register(xterm.kill)
    slave = os.fdopen(slave, 'r+')
    slave.readline() # Get rid of xterm ID
    pdb.Pdb(stdin=slave, stdout=slave).set_trace()