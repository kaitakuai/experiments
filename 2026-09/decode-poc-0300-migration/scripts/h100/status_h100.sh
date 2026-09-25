#!/usr/bin/env bash
chmod +x /root/dl_glm.sh 2>/dev/null
if ! pgrep -f 'dl_gl[m]' >/dev/null; then nohup /root/dl_glm.sh > /root/dl_glm.log 2>&1 & echo "glm download started"; fi
sleep 5
echo "minimax $(du -sh /root/models/MiniMax-M2.7 2>/dev/null | cut -f1) $(grep -h done /root/dl_minimax.log 2>/dev/null | cut -c1-40)"
echo "glm $(du -sh /root/models/GLM-5.3-Flash 2>/dev/null | cut -f1) $(tail -c 120 /root/dl_glm.log 2>/dev/null | tr '\r' '\n' | tail -1 | cut -c1-80)"
echo "venv: $(tail -1 /root/mkvenv.log | cut -c1-100)"
echo "ref: $(du -sh /root/ref 2>/dev/null | cut -f1); free $(df -h / | tail -1 | awk '{print $4}')"
if grep -q 'venv done rc=0' /root/mkvenv.log && [ -f /root/setup030.sh ] && [ ! -f /root/setup030.done ]; then bash /root/setup030.sh 2>&1 | grep -v '^INFO\|^WARNING\|^\[' | tail -4 && touch /root/setup030.done; fi
