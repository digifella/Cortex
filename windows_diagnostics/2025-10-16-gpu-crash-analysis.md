## Windows GPU Crash Analysis – 2025-10-16 17:27

### Timeline
- **17:24:13** – Application log records multiple `Windows Error Reporting` `LiveKernelEvent` **141** entries (TDR watchdog). Dump files stored under `C:\WINDOWS\LiveKernelReports\WATCHDOG\WATCHDOG-20251016-1043.dmp` (and related temp XML/CSV metadata).
- **17:27:02** – System log `Kernel-Power` **Event 41 (Critical)** indicates an unexpected reboot (hard reset / power loss scenario).
- **17:27:32** – System log `Kernel-PnP` **Event 219 (Warning)** shows `\Driver\WudfRd` failing to load for display device `DISPLAY\SHP1587\4&12949b5&0&UID8388688` with status `0xC0000365`.

### Interpretation
- Cluster of `LiveKernelEvent 141` entries suggests the GPU hung and triggered Windows Timeout Detection and Recovery (TDR).
- Follow-up Event 41 confirms the machine rebooted without a clean shutdown, consistent with the GPU hang watchdog resetting the system.
- Event 219 immediately after reboot indicates the display driver stack remained unstable during device initialization.
- Overall: evidence supports a graphics driver / GPU overload leading to TDR and forced reboot.

### Files Worth Inspecting Later
- `C:\WINDOWS\LiveKernelReports\WATCHDOG\WATCHDOG-20251016-1043.dmp`
- Corresponding WER temp files listed in Application log (for richer metadata).

### Next Steps (Planned)
1. Install latest NVIDIA GPU drivers (currently in progress).
2. After reboot, run `WinDbg` on the WATCHDOG dump (`!analyze -v`) to verify the exact driver/module causing the TDR.
3. Check GPU thermals and utilization under load (e.g., HWInfo, GPU-Z, or `nvidia-smi`) to rule out overheating.
4. Stress-test GPU (FurMark / 3DMark loop) once stable to confirm no further LiveKernelEvent 141 entries.
5. Review Windows Reliability Monitor for additional context or repeated patterns around the crash timeframe.
