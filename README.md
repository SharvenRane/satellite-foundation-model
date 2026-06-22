# satellite-foundation-model

A small but complete example of self supervised pretraining for satellite
imagery, followed by linear probing on a downstream task. The encoder is a
masked autoencoder that learns from unlabeled multispectral patches. Once it is
pretrained, its frozen features are used to fit a simple classifier, and that
classifier is compared against the same probe run on an untrained encoder so the
benefit of pretraining is measured honestly.

Everything is synthetic and runs on CPU in a few seconds, so the whole pipeline
is easy to read, run, and test without any downloads or accounts.

## The idea

Foundation models for remote sensing are usually pretrained on huge piles of
unlabeled imagery and then adapted to specific tasks with a small head. This
repo reproduces that shape in miniature:

1. Generate synthetic multispectral patches. Each patch has a faint, coherent,
   low frequency pattern whose orientation depends on its class, buried under
   strong class independent sensor noise. The per band offsets and the noise are
   randomized, and every band is normalized to zero mean and unit variance, so
   the label cannot be read off simple statistics like brightness or energy. The
   only reliable cue is the recoverable spatial structure.
2. Pretrain a masked autoencoder on these patches with no labels. The patch is
   split into a grid of tiles, most of the tiles are hidden, and the model has
   to reconstruct the missing pixels from the visible ones. To do that well it
   has to average the noise away and represent the coherent structure.
3. Probe the result. Freeze the encoder, pool its features, and fit a logistic
   regression classifier on a labeled set. Run the very same probe on a randomly
   initialized encoder for a baseline, and report both next to the chance level.

The point of the noise is that a random encoder mostly passes the noise through,
so its features are weak, while the pretrained encoder has learned to surface the
signal. That gap is what the tests check.

## Architecture

The model in `src/model.py` is a real, if small, masked autoencoder.

- `PatchEmbed` splits a multiband patch into non overlapping tiles and linearly
  embeds each tile. Its `tilify` and `untilify` methods are exact inverses, so
  reconstruction targets line up with predictions.
- `MAEEncoder` adds fixed sinusoidal positional embeddings and runs a stack of
  transformer blocks. During pretraining it sees only the visible tiles, which
  is the efficiency trick that makes masked autoencoding cheap.
- `MAEDecoder` scatters the encoded visible tiles back into place, fills the
  hidden positions with a learned mask token, and reconstructs every tile.
- The loss is mean squared error computed only over the hidden tiles, the
  standard masked autoencoding objective.

`LinearProbe` is a single linear layer, but the probe used in the tests and the
demo is a logistic regression from scikit-learn fit on pooled features, which is
the usual way to evaluate a frozen representation.

## Layout

```
src/
  data.py       synthetic multispectral patch generator
  model.py      patch embedding, encoder, decoder, full MAE, linear probe
  pretrain.py   the self supervised training loop and feature extraction
  probe.py      linear probing and the chance level baseline
  demo.py       end to end script you can run directly
tests/
  test_data.py      shapes, reproducibility, class separability
  test_model.py     tiling roundtrip, masking consistency, gradient flow
  test_pretrain.py  loss decreases, training is reproducible
  test_probe.py     pretrained features beat chance and beat random features
```

## Running it

Install the dependencies and run the demo:

```
pip install -r requirements.txt
python -m src.demo
```

On one run the demo printed:

```
masked reconstruction loss
  first epoch: 1.1963
  last epoch:  0.9279

linear probe test accuracy
  chance level:        0.283
  random encoder:      0.383
  pretrained encoder:  1.000
```

The reconstruction loss falls over training, and the pretrained encoder's
features classify the held out patches far better than both the chance level and
a randomly initialized encoder. These are numbers from an actual run on this
synthetic data, not a benchmark claim about real imagery.

## Results on real imagery (EuroSAT)

`src/eurosat.py` swaps the synthetic generator for EuroSAT, 27,000 real
Sentinel-2 RGB patches across 10 land use classes, auto downloaded through
torchvision. After MAE pretraining on an RTX 5070 Ti, the frozen feature linear
probe reached 0.560 test accuracy, against 0.351 for a randomly initialized
encoder and a 0.115 chance level. The gap over the random encoder is the real
signal: pretraining on unlabeled patches learned structure a linear probe can
read off. Modest in absolute terms, as expected for a small MAE on a short
schedule, but a real and reproducible result on real imagery.

## Tests

```
pytest tests/ -q
```

The tests are behavior checks rather than fixed number assertions. They confirm
that the tiling is lossless, that the masking produces a valid permutation and a
consistent visible set, that gradients reach the encoder, that the pretraining
loss drops by a clear margin, that a seeded run is reproducible, and that the
probed pretrained features beat both chance and the random encoder baseline. All
of them run on CPU with tiny tensors.

## Notes and limitations

The data is synthetic by design so the repo stays self contained and fast. The
absolute accuracy numbers are a property of that easy synthetic task and say
nothing about real Sentinel or Landsat imagery. What carries over is the
structure: an unlabeled masked autoencoding objective, a frozen feature probe,
and an honest baseline to compare against. To move toward real data you would
swap the generator in `data.py` for a loader of real multispectral tiles, scale
up the encoder depth and width, and keep the same pretraining and probing loops.
