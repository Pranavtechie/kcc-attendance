## Important Tasks

- [ ] Setting up Flask Server
- [ ] Setting up gRPC
- [ ] Updating UI a LOT

## Final Checklist

- [x] Convert `.onnx` model weights to `.rknn`
- [x] Setup proper code for inference from `.rknn` model weights, when rockchip compatible system is detected
- [ ] Command for NPU Usage => `watch sudo cat /sys/kernel/debug/rknpu/load`, find way to store the timestamp and the usage properly using prometheus
- [ ] Telemetry and tracking of NPU / GPU usage, CPU using prometheus etc ???
- [ ] Hibernation Mode => if camera black -> enter hibernation mode / check every 5 sec (capturing new frame)

## Whimsical / Unimportant

- [ ] Animations in pyside6
- [ ] Use empty space to display questions / ascii art / image etc.
