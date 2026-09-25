#!/usr/bin/env bash
echo "start $(date -u +%T)"
/root/hf/bin/hf download deepseek-ai/DeepSeek-V4-Flash-0731 --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 --local-dir /root/models/DeepSeek-V4-Flash-0731 --max-workers 16
echo "deepseek fp8 done rc=$? $(date -u +%T)"
