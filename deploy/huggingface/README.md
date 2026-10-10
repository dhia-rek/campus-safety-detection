---
title: Campus Safety Detection (demo)
emoji: 🛡️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Zero-shot CCTV incident detection demo on synthetic data
---

# Campus Safety Detection: demo

Dashboard and agent layer of [dhia-rek/campus-safety-detection](https://github.com/dhia-rek/campus-safety-detection)
running on **synthetic** stick-figure scenes. Score curves are synthetic, not model output;
alerts go to a mock Telegram notifier. No camera, GPU or token is involved.

Detection and alerting on real footage is a human-in-the-loop aid, not an automatic judge. See the
repository README for limitations and the ethics and privacy section.
