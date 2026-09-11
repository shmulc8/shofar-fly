# Shofar Fly

A real fruit-fly brain learns to blow the shofar. Rosh Hashanah 5787.

The brain is the FlyWire FAFB connectome (139,255 neurons, 2.7M synapses) simulated as leaky integrate-and-fire neurons in a Web Worker. A sound cue is played into the fly's ear neurons (Johnston's organ). The readout is the fly's own mouth motor neurons. Training uses dopamine-gated three-factor plasticity on the 1,231 excitatory synapses from the cue pathway onto the mouth motor neurons: synapses active while the mouth answers the sound get stronger when dopamine arrives.

The shofar sound is driven only by the measured mouth motor spike rate in each trial. Untrained: about 3.6 spikes per tick, a sputter. Trained: about 5.2, a clean blast.

## Run

```
python3 -m http.server 8765
```

Open http://127.0.0.1:8765/stage.html and wait for the connectome to load.

- **S** train from scratch in fast-forward (about 45 s), then the shofar service starts automatically
- **B** blow the ten-blast service with the current brain (TaShRaT, TaShaT, TaRaT, ending in a tekiah gedolah)
- **H** hide the HUD
- **T** toggle the pre-trained weights on and off

## Training offline

`tools/train_gpu.py` mirrors the browser simulator in PyTorch (Apple MPS or CPU) and runs the conditioning protocol with controls: paired reward, backward reward, unpaired reward, no reward. It writes `stage/trained_weights.bin`, which the page loads at start.

```
python -m venv venv && venv/bin/pip install torch numpy
venv/bin/python tools/train_gpu.py --parallel --trials 200 --no-tonic --gain 150 --adapt-inc 1.0 --adapt-decay 0.97 --da0 0 --eta 0.002 --reset-between --plastic-post SEZ_FEED,MN_PROBOSCIS,MN_HEAD --cue-path --da-mode signal --arms pair,backward,unpaired,noreward
```

Last result (cue-evoked mouth spikes per tick, dopamine-free test trials): paired 3.60 to 5.60, backward 3.60 to 3.90, unpaired 3.60 to 3.60, no reward 3.60 to 3.60.

## Credits

- Simulator and connectome pipeline: [snedea/flybrain](https://github.com/snedea/flybrain), MIT (see LICENSE-flybrain.md)
- Connectome: FlyWire FAFB v783
- Fly mesh: decimated from [flybody](https://github.com/TuragaLab/flybody), built with `tools/build_fly_mesh.py`
- 3D: three.js r128
