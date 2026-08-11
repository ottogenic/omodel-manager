#!/bin/sh
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from NVIDIA/dgx-spark-playbooks commit
# 1fb66f059ee427c5a3678b3117ef73aab042b458.
set -eu

SSH_PORT="${SSH_PORT:-2233}"
SSH_LISTEN_ADDRESS="${SSH_LISTEN_ADDRESS:?SSH_LISTEN_ADDRESS must be set}"
mkdir -p /var/run/sshd /root/.ssh
cp -R /tmp/.ssh/. /root/.ssh/
chown -R root:root /root/.ssh
chmod 700 /root/.ssh
find /root/.ssh -type f -name '*.pub' -exec chmod 644 {} \;
find /root/.ssh -type f ! -name '*.pub' -exec chmod 600 {} \;

sed -i.bak \
    -e 's/^#\?\s*PermitRootLogin\s.*/PermitRootLogin yes/' \
    -e 's/^#\?\s*PubkeyAuthentication\s.*/PubkeyAuthentication yes/' \
    -e "s/^#\?\s*Port\s\+22\s*$/Port ${SSH_PORT}/" \
    /etc/ssh/sshd_config
printf '\nListenAddress %s\n' "$SSH_LISTEN_ADDRESS" >> /etc/ssh/sshd_config
printf '\nHost *\n    StrictHostKeyChecking no\n    Port %s\n    UserKnownHostsFile=/dev/null\n' \
    "$SSH_PORT" > /etc/ssh/ssh_config.d/trt-llm.conf
chmod 600 /etc/ssh/ssh_config.d/trt-llm.conf
sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' \
    -i /etc/pam.d/sshd

exec /usr/sbin/sshd -D
