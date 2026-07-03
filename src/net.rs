//! Minimal dense neural network (ReLU MLP, weighted-MSE loss, Adam) —
//! dependency-free and deterministic, used for the belief-state value
//! network. Gradients are verified against finite differences in the tests.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// (input, target, per-output loss weight)
pub type Example = (Vec<f32>, Vec<f32>, Vec<f32>);

#[derive(Clone, Serialize, Deserialize)]
pub struct Mlp {
    pub sizes: Vec<usize>,
    /// w[l]: sizes[l+1] × sizes[l], row-major.
    pub w: Vec<Vec<f32>>,
    pub b: Vec<Vec<f32>>,
}

pub struct Grads {
    w: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
}

impl Grads {
    fn zeros(net: &Mlp) -> Grads {
        Grads {
            w: net.w.iter().map(|w| vec![0.0; w.len()]).collect(),
            b: net.b.iter().map(|b| vec![0.0; b.len()]).collect(),
        }
    }

    fn add(&mut self, o: &Grads) {
        for (a, b) in self.w.iter_mut().zip(&o.w) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        for (a, b) in self.b.iter_mut().zip(&o.b) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
    }

    fn scale(&mut self, k: f32) {
        for a in &mut self.w {
            for x in a {
                *x *= k;
            }
        }
        for a in &mut self.b {
            for x in a {
                *x *= k;
            }
        }
    }
}

impl Mlp {
    /// He-uniform initialization.
    pub fn new(sizes: &[usize], seed: u64) -> Mlp {
        assert!(sizes.len() >= 2);
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut w = Vec::new();
        let mut b = Vec::new();
        for l in 0..sizes.len() - 1 {
            let (fan_in, fan_out) = (sizes[l], sizes[l + 1]);
            let bound = (6.0 / fan_in as f32).sqrt();
            w.push(
                (0..fan_in * fan_out)
                    .map(|_| rng.random_range(-bound..bound))
                    .collect(),
            );
            b.push(vec![0.0; fan_out]);
        }
        Mlp {
            sizes: sizes.to_vec(),
            w,
            b,
        }
    }

    /// Layer activations: `acts[0]` = input, ReLU on hidden layers, linear
    /// output.
    fn acts(&self, x: &[f32]) -> Vec<Vec<f32>> {
        debug_assert_eq!(x.len(), self.sizes[0]);
        let layers = self.w.len();
        let mut acts = Vec::with_capacity(layers + 1);
        acts.push(x.to_vec());
        for l in 0..layers {
            let n_in = self.sizes[l];
            let prev = &acts[l];
            let mut out = self.b[l].clone();
            let w = &self.w[l];
            for (o, out_o) in out.iter_mut().enumerate() {
                let row = &w[o * n_in..(o + 1) * n_in];
                let mut acc = 0.0f32;
                for (v, p) in row.iter().zip(prev) {
                    acc += v * p;
                }
                *out_o += acc;
            }
            if l + 1 < layers {
                for v in &mut out {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            acts.push(out);
        }
        acts
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.acts(x).pop().unwrap()
    }

    /// Weighted-MSE loss `Σ wt[o]·(y−t)²` and its gradients for one example.
    fn example_grads(&self, x: &[f32], target: &[f32], wt: &[f32]) -> (Grads, f32) {
        let acts = self.acts(x);
        let layers = self.w.len();
        let y = &acts[layers];
        let mut loss = 0.0f32;
        let mut delta: Vec<f32> = y
            .iter()
            .zip(target)
            .zip(wt)
            .map(|((&y, &t), &w)| {
                let d = y - t;
                loss += w * d * d;
                2.0 * w * d
            })
            .collect();

        let mut g = Grads::zeros(self);
        for l in (0..layers).rev() {
            let (n_in, n_out) = (self.sizes[l], self.sizes[l + 1]);
            let prev = &acts[l];
            let gw = &mut g.w[l];
            for o in 0..n_out {
                let d = delta[o];
                if d != 0.0 {
                    let row = &mut gw[o * n_in..(o + 1) * n_in];
                    for (r, p) in row.iter_mut().zip(prev) {
                        *r += d * p;
                    }
                }
            }
            g.b[l].copy_from_slice(&delta);
            if l > 0 {
                let w = &self.w[l];
                let mut prev_delta = vec![0.0f32; n_in];
                for (o, &d) in delta.iter().enumerate() {
                    if d != 0.0 {
                        let row = &w[o * n_in..(o + 1) * n_in];
                        for (pd, &wv) in prev_delta.iter_mut().zip(row) {
                            *pd += d * wv;
                        }
                    }
                }
                // ReLU gate.
                for (pd, &a) in prev_delta.iter_mut().zip(prev) {
                    if a <= 0.0 {
                        *pd = 0.0;
                    }
                }
                delta = prev_delta;
            }
        }
        (g, loss)
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let bytes = bincode::serialize(self).map_err(std::io::Error::other)?;
        std::fs::write(path, bytes)
    }

    pub fn load(path: &str) -> std::io::Result<Mlp> {
        let bytes = std::fs::read(path)?;
        bincode::deserialize(&bytes).map_err(std::io::Error::other)
    }
}

pub struct Adam {
    lr: f32,
    t: i32,
    m: Grads,
    v: Grads,
}

const B1: f32 = 0.9;
const B2: f32 = 0.999;
const EPS: f32 = 1e-8;

impl Adam {
    pub fn new(net: &Mlp, lr: f32) -> Adam {
        Adam {
            lr,
            t: 0,
            m: Grads::zeros(net),
            v: Grads::zeros(net),
        }
    }

    pub fn step(&mut self, net: &mut Mlp, g: &Grads) {
        self.t += 1;
        let bc1 = 1.0 - B1.powi(self.t);
        let bc2 = 1.0 - B2.powi(self.t);
        let lr = self.lr * bc2.sqrt() / bc1;
        for l in 0..net.w.len() {
            for (i, &gw) in g.w[l].iter().enumerate() {
                let m = &mut self.m.w[l][i];
                let v = &mut self.v.w[l][i];
                *m = B1 * *m + (1.0 - B1) * gw;
                *v = B2 * *v + (1.0 - B2) * gw * gw;
                net.w[l][i] -= lr * *m / (v.sqrt() + EPS);
            }
            for (i, &gb) in g.b[l].iter().enumerate() {
                let m = &mut self.m.b[l][i];
                let v = &mut self.v.b[l][i];
                *m = B1 * *m + (1.0 - B1) * gb;
                *v = B2 * *v + (1.0 - B2) * gb * gb;
                net.b[l][i] -= lr * *m / (v.sqrt() + EPS);
            }
        }
    }
}

/// One epoch of shuffled minibatch training (gradients averaged over the
/// batch, computed in parallel). Returns the mean per-example loss.
pub fn train_epoch(
    net: &mut Mlp,
    adam: &mut Adam,
    data: &[Example],
    batch: usize,
    rng: &mut SmallRng,
) -> f32 {
    let mut idx: Vec<usize> = (0..data.len()).collect();
    for k in 0..idx.len() {
        let j = rng.random_range(k..idx.len());
        idx.swap(k, j);
    }
    let mut total_loss = 0.0f64;
    for chunk in idx.chunks(batch) {
        let (grads, loss) = chunk
            .par_iter()
            .map(|&i| {
                let (x, t, wt) = &data[i];
                net.example_grads(x, t, wt)
            })
            .reduce_with(|(mut ga, la), (gb, lb)| {
                ga.add(&gb);
                (ga, la + lb)
            })
            .unwrap();
        let mut grads = grads;
        grads.scale(1.0 / chunk.len() as f32);
        adam.step(net, &grads);
        total_loss += loss as f64;
    }
    (total_loss / data.len() as f64) as f32
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Analytic gradients must match central finite differences on a small
    /// random net.
    #[test]
    fn gradients_match_finite_differences() {
        let mut rng = SmallRng::seed_from_u64(11);
        let net = Mlp::new(&[5, 7, 4], 3);
        let x: Vec<f32> = (0..5).map(|_| rng.random_range(-1.0..1.0)).collect();
        let t: Vec<f32> = (0..4).map(|_| rng.random_range(-1.0..1.0)).collect();
        let wt: Vec<f32> = (0..4).map(|_| rng.random_range(0.1..1.0)).collect();
        let (g, _) = net.example_grads(&x, &t, &wt);

        let eps = 3e-3f32;
        let mut checked = 0;
        for l in 0..net.w.len() {
            for i in (0..net.w[l].len()).step_by(5) {
                let mut plus = net.clone();
                plus.w[l][i] += eps;
                let mut minus = net.clone();
                minus.w[l][i] -= eps;
                let lp: f32 = plus
                    .forward(&x)
                    .iter()
                    .zip(&t)
                    .zip(&wt)
                    .map(|((&y, &t), &w)| w * (y - t) * (y - t))
                    .sum();
                let lm: f32 = minus
                    .forward(&x)
                    .iter()
                    .zip(&t)
                    .zip(&wt)
                    .map(|((&y, &t), &w)| w * (y - t) * (y - t))
                    .sum();
                let fd = (lp - lm) / (2.0 * eps);
                assert!(
                    (g.w[l][i] - fd).abs() < 2e-2 * (1.0 + fd.abs()),
                    "grad mismatch at layer {l} idx {i}: analytic {} vs fd {}",
                    g.w[l][i],
                    fd
                );
                checked += 1;
            }
        }
        assert!(checked > 10);
    }

    /// A small net must be able to memorize a tiny dataset (sanity of the
    /// full training loop: shuffling, batching, Adam).
    #[test]
    fn overfits_a_tiny_dataset() {
        let mut rng = SmallRng::seed_from_u64(21);
        let data: Vec<Example> = (0..24)
            .map(|_| {
                let x: Vec<f32> = (0..6).map(|_| rng.random_range(-1.0..1.0)).collect();
                // Nonlinear target the net has to actually learn.
                let t = vec![
                    x[0] * x[1] + x[2],
                    (x[3] - x[4]).max(0.0),
                    x[5].abs(),
                ];
                (x, t, vec![1.0; 3])
            })
            .collect();

        let mut net = Mlp::new(&[6, 48, 48, 3], 7);
        let mut adam = Adam::new(&net, 3e-3);
        let mut loss = f32::MAX;
        for _ in 0..800 {
            loss = train_epoch(&mut net, &mut adam, &data, 8, &mut rng);
        }
        assert!(loss < 1e-3, "failed to overfit: final loss {loss}");
    }

    #[test]
    fn save_load_roundtrip() {
        let net = Mlp::new(&[4, 8, 2], 5);
        let dir = std::env::temp_dir().join("pluribus_net_test.bin");
        let path = dir.to_str().unwrap();
        net.save(path).unwrap();
        let back = Mlp::load(path).unwrap();
        let x = vec![0.3, -0.2, 0.8, 0.1];
        assert_eq!(net.forward(&x), back.forward(&x));
        let _ = std::fs::remove_file(path);
    }
}
