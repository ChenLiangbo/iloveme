#!/usr/bin/env bash

PROJDIR="`pwd`"
PIDFILE="`pwd`/alertapp.pid"

exec /usr/bin/env python manage.py runfcgi method=prefork host=127.0.0.1 port=3820 pidfile=$PIDFILE
