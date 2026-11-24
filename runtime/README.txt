runtime/win64/msvcp140.dll
==========================

Microsoft Visual C++ runtime library (msvcp140.dll, version 14.51.36247.0, signed by Microsoft
Corporation), deployed app-locally for the offline chatbot.

Why: llama-cpp-python's Windows DLLs are built with a recent MSVC toolset. This computer's system copy
(C:\Windows\System32\msvcp140.dll, version 14.34) is older than 14.40 and makes those builds crash with an
access violation. Python does not ship msvcp140.dll, so the chatbot pre-loads this copy
(src/chatbot/runtime.py). Nothing is installed or changed system-wide.

Source: the same Microsoft-signed file already used app-locally by the portable MongoDB server
(D:\tools\mongodb\server-9.0.4\bin). Microsoft permits app-local deployment of the Visual C++
redistributable files.

If the system runtime is ever updated to 14.40 or newer, this file is no longer used.


runtime/gpu/<backend>/
======================

GPU builds of llama-cpp-python (same version as the CPU build in .venv), installed by
`python scripts/setup.py --chatbot` from requirements-gpu-vulkan.txt (default) and, optionally,
requirements-gpu-cuda.txt. They are kept outside .venv so the CPU build stays intact as the fallback.
src/chatbot/device.py tests them at start-up and uses the first one that works on a GPU; otherwise the
CPU build is used. Deleting this folder is safe: the chatbot then always runs on the CPU.
