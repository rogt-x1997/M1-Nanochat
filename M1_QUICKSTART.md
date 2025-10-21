# M1 MacBook Quick Start Guide

This guide is specifically optimized for **MacBook Air M1** users who want to train an LLM locally in under 1 hour.

## What's Optimized for M1?

Your MacBook Air M1 has:
- **8-core CPU** (4 performance + 4 efficiency cores)
- **7-8 core GPU** with Metal Performance Shaders (MPS)
- **8-16 GB unified memory**

The pipeline automatically uses **MPS acceleration** for ~3-5x faster training compared to CPU-only.

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install torch numpy streamlit
```

That's it! The M1 version has minimal dependencies.

### Step 2: Choose Your Method

#### Option A: Streamlit UI (Recommended - Easiest)

```bash
streamlit run app_m1.py
```

Then in your browser:
1. Choose a preset: **"Balanced (~30 min)"**
2. Click **"🚀 Start Training"**
3. Wait ~30 minutes
4. Generate text in the **"Inference"** tab

#### Option B: Python Script (Automated)

```bash
python train_m1.py
```

This will:
- Prepare data automatically
- Train a 6-layer model (3M parameters)
- Take ~30-45 minutes
- Generate sample text
- Save the model

### Step 3: Generate Text

After training, use the model to generate text!

## Configuration Presets

Choose based on your time budget:

| Preset | Time | Layers | Params | Quality |
|--------|------|--------|--------|---------|
| **Fast** | ~15 min | 4 | ~1M | Good for testing |
| **Balanced** | ~30 min | 6 | ~3M | Best value ⭐ |
| **Quality** | ~45 min | 8 | ~6M | Best quality |

**Recommendation:** Start with **Balanced** preset for the best time/quality tradeoff.

## What to Expect

### Training Progress

You'll see output like this:

```
Iter    0 | Train Loss: 3.45 | Val Loss: 3.42 | Time: 0.0s
Iter  500 | Train Loss: 2.10 | Val Loss: 2.15 | Time: 3.2m
Iter 1000 | Train Loss: 1.65 | Val Loss: 1.72 | Time: 6.5m
Iter 5000 | Train Loss: 1.20 | Val Loss: 1.35 | Time: 32.0m
```

**Good signs:**
- Loss decreasing steadily
- Validation loss close to training loss
- ~150-200 iterations per minute on M1

### Generated Text Quality

After 5000 iterations (~30 min), expect:
- **Coherent sentences** ✅
- **Basic story structure** ✅
- **Some repetition** (normal for small models)
- **Occasional nonsense** (improves with more training)

Example output:
```
Prompt: "Once upon a time"
Generated: "Once upon a time there was a little girl who loved 
to play in the forest. She found a magical stone and became 
friends with a fairy..."
```

## Performance Tips

### 1. Close Other Apps
Free up memory by closing:
- Chrome/Safari (or keep only 1-2 tabs)
- Slack, Discord, etc.
- Other heavy applications

### 2. Monitor Activity
Open **Activity Monitor** to check:
- **Memory pressure** should be green/yellow
- **GPU usage** should show ~80-100% during training
- **CPU usage** will be moderate (20-40%)

### 3. Optimal Settings for M1 Air

```json
{
  "batch_size": 16,        // Good balance for 8GB RAM
  "seq_length": 256,       // Reasonable context
  "n_layer": 6,            // Sweet spot for M1
  "n_embd": 384,           // Good capacity
  "max_iters": 5000        // ~30 minutes
}
```

### 4. If You Run Out of Memory

Reduce these settings:
1. **Batch size**: 16 → 12 → 8
2. **Sequence length**: 256 → 128
3. **Hidden dim**: 384 → 256
4. **Layers**: 6 → 4

## Troubleshooting

### "MPS not available"

If you see this warning, the pipeline falls back to CPU (slower but still works).

**Fix:**
- Update to macOS 12.3+ (required for MPS)
- Update PyTorch: `pip install --upgrade torch`

### Training is Slow

**Expected speeds on M1 Air:**
- With MPS: ~150-200 iters/min
- CPU only: ~30-50 iters/min

**If slower:**
- Close other apps
- Check Activity Monitor for memory pressure
- Reduce batch size

### Out of Memory Error

**Solutions:**
1. Reduce batch size to 8
2. Reduce sequence length to 128
3. Use "Fast" preset instead of "Balanced"
4. Close other applications

### Generated Text is Gibberish

**This is normal if:**
- Training just started (< 1000 iterations)
- Model is very small (< 1M parameters)

**Improve quality:**
- Train longer (10,000+ iterations)
- Use "Quality" preset
- Lower temperature (0.5-0.7) during generation

## Advanced: Custom Training

Edit `configs/m1_macbook_config.json` to customize:

```json
{
  "model": {
    "n_layer": 8,          // More layers = better quality
    "n_embd": 512,         // Larger = more capacity
    "sequence_len": 256    // Longer context
  },
  "training": {
    "max_iters": 10000,    // More iterations = better
    "batch_size": 12,      // Adjust for your RAM
    "learning_rate": 3e-4  // Usually don't change
  }
}
```

Then run:
```bash
python train_m1.py
```

## Time Estimates (M1 Air)

| Iterations | Time | Expected Loss |
|------------|------|---------------|
| 1,000 | ~5 min | ~2.0 |
| 2,000 | ~10 min | ~1.7 |
| 5,000 | ~30 min | ~1.3 |
| 10,000 | ~60 min | ~1.1 |

**Note:** Times are for the "Balanced" preset (6 layers, 384 dim, batch 16)

## Next Steps After Training

1. **Save your model** in the Streamlit UI
2. **Experiment with generation settings**:
   - Temperature: 0.7-0.9 for creative text
   - Top-k: 20-50 for diversity
   - Top-p: 0.9 for quality
3. **Train on your own data**:
   - Paste custom text in the UI
   - Or modify `train_m1.py` to load your files
4. **Train longer** for better quality:
   - Change `max_iters` to 10,000 or 20,000
   - Will take 1-2 hours but much better results

## Example Workflow

```bash
# 1. Start training (30 minutes)
streamlit run app_m1.py
# Choose "Balanced" preset → Click "Start Training"

# 2. While training, grab coffee ☕
# Watch the loss decrease in real-time

# 3. After training, generate text
# Go to "Inference" tab
# Try different prompts and settings

# 4. Save your model
# Click "Save Model" button

# 5. Share your results! 🎉
```

## FAQ

**Q: Can I use my M1 MacBook Pro instead?**  
A: Yes! It will be even faster due to better cooling and sustained performance.

**Q: What about M2/M3 Macs?**  
A: Works great! May be 20-30% faster than M1.

**Q: Can I train overnight?**  
A: Yes! Set `max_iters` to 50,000+ for very high quality. Will take 4-6 hours.

**Q: How does this compare to Colab?**  
A: M1 MPS ≈ Colab T4 GPU in speed. But you have unlimited time!

**Q: Can I train larger models?**  
A: On 8GB RAM, stick to < 10M parameters. On 16GB RAM, you can go up to ~30M parameters.

---

**Ready to start?** Run `streamlit run app_m1.py` and begin training! 🚀

