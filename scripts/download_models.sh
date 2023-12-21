#!/usr/bin/env bash
case $(pwd) in *scripts) cd ..;; esac
curl -L -o models.tar.gz https://www.atarismwc.com/models.tar.gz
tar -xvzf models.tar.gz
rm models.tar.gz

