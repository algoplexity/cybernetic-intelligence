# V4 Architecture Implementation Plan

## Critical Implementation Details

### 1. **PermutationSymbolizer** - The Foundation
```python
class PermutationSymbolizer:
    def __init__(self, embedding_dim, seed):
        # Key insight: Use ordinal patterns to capture local dynamics
        # This makes the approach invariant to amplitude scaling
        self.embedding_dim = embedding_dim
        self.rng = np.random.RandomState(seed)
    
    def symbolize_vector(self, vector):
        # Convert to ordinal pattern (Bandt-Pompe symbolization)
        # Handle ties with randomized tie-breaking for robustness
        return self._compute_ordinal_pattern(vector)
```

**Why this works**: Ordinal patterns capture the relative ordering of values, making the approach robust to noise and amplitude variations while preserving temporal structure.

### 2. **HierarchicalDynamicalEncoder** - The Core Innovation
```python
class HierarchicalDynamicalEncoder(nn.Module):
    def forward(self, sequence_batch):
        # CRITICAL: Must return (fingerprint_sequence, residuals_list)
        # The residuals enable perfect reconstruction in the decoder
        fingerprint_seq, residuals = self.encode_hierarchically(sequence_batch)
        return fingerprint_seq, residuals
```

**Key Design Decision**: The tuple return format ensures that the decoder can perfectly reconstruct the input, which is essential for the MDL objective.

### 3. **MDL_AU_Net_Autoencoder** - The Pre-training Model
```python
class MDL_AU_Net_Autoencoder(nn.Module):
    def forward(self, sequence_batch):
        # Encode with residuals
        fingerprint_seq, residuals = self.encoder(sequence_batch)
        
        # Decode for reconstruction
        reconstructed = self.decoder(fingerprint_seq, residuals)
        
        # Classify for rule identification
        rule_logits = self.classifier(fingerprint_seq)
        
        return reconstructed, rule_logits
```

**MDL Objective**: The model learns to compress (encode) and decompress (decode) while maintaining the ability to classify the underlying rule. This forces it to learn meaningful, structured representations.

### 4. **StructuralBreakClassifier** - The Fine-tuning Model
```python
class StructuralBreakClassifier(nn.Module):
    def forward(self, before_seqs, after_seqs):
        # Process both periods
        before_fingerprints = [self.encoder(seq)[0] for seq in before_seqs]  # [0] extracts fingerprint
        after_fingerprints = [self.encoder(seq)[0] for seq in after_seqs]
        
        # Average fingerprints for stability
        avg_before = torch.stack(before_fingerprints).mean(dim=0)
        avg_after = torch.stack(after_fingerprints).mean(dim=0)
        
        # Compare fingerprints
        return self.classifier(torch.cat([avg_before, avg_after], dim=-1))
```

**Key Insight**: Averaging multiple fingerprints from the same period increases robustness to noise and provides a more stable representation.

## Implementation Priorities

### Phase 1: Core Components (Week 1-2)
1. **PermutationSymbolizer**
   - Implement ordinal pattern computation
   - Add robust tie-breaking
   - Validate on synthetic data

2. **SeriesProcessor**
   - Time-delay embedding
   - Sliding window extraction
   - Edge case handling

3. **Basic Encoder-Decoder**
   - Simple transformer blocks
   - Residual connections
   - Tuple return format

### Phase 2: Advanced Architecture (Week 2-3)
1. **Hierarchical Attention**
   - Multi-scale processing
   - Skip connections
   - Residual preservation

2. **ECA Data Generation**
   - Diverse rule synthesis
   - Composite rule handling
   - Balanced dataset creation

3. **MDL Training Loop**
   - Reconstruction loss
   - Classification loss
   - Proper loss weighting

### Phase 3: Fine-tuning and Optimization (Week 3-4)
1. **Structural Break Classifier**
   - Fingerprint averaging
   - Comparison mechanisms
   - Calibration for probability output

2. **Pipeline Integration**
   - End-to-end training
   - Model persistence
   - Inference optimization

## Critical Success Factors

### 1. **Representation Quality**
- The encoder must learn meaningful, transferable representations
- Test on diverse synthetic datasets before real data
- Validate that similar dynamics produce similar fingerprints

### 2. **Stability and Robustness**
- Averaging multiple fingerprints is crucial for noisy data
- Proper normalization at each stage
- Robust handling of edge cases (short series, missing values)

### 3. **Computational Efficiency**
- Model size must fit competition constraints
- Inference time must be reasonable
- Memory usage optimization for long sequences

## Validation Strategy

### Synthetic Data Tests
1. **ECA Rule Transitions**: Test on known rule changes
2. **Noise Robustness**: Add varying levels of noise
3. **Scale Invariance**: Test with different amplitude scales
4. **Temporal Robustness**: Vary sequence lengths

### Real Data Validation
1. **Cross-validation**: Split training data properly
2. **Ablation Studies**: Test individual components
3. **Comparison**: Benchmark against statistical methods
4. **Interpretability**: Analyze learned representations

## Risk Mitigation

### Technical Risks
- **Overfitting to ECA**: Use diverse synthetic data
- **Poor Transfer**: Validate on held-out real data early
- **Computational Cost**: Profile and optimize bottlenecks

### Architectural Risks
- **Tuple Format Issues**: Extensive unit testing
- **Fingerprint Averaging**: Validate mathematical correctness
- **Loss Function Balance**: Systematic hyperparameter search

## Expected Advantages Over Baseline

1. **Temporal Context**: Captures sequential dependencies
2. **Learned Representations**: Adapts to data characteristics
3. **Multi-scale Processing**: Handles different break timescales
4. **Robustness**: Ordinal patterns + averaging increase stability
5. **Transferability**: Pre-trained representations generalize better
