"""Two-level UNet velocity field for Darcy-flow triangular transport.

Architecture matches the `DirectFlowUNet` from the reference image-aux
codebase, adapted for the 33×33 Poisson parameter-estimation problem.

Classes
-------
DarcyUNet : eqx.Module
    Two-level UNet. Signature: (t, y_flat, u_flat) -> v_flat.
    - t      scalar in [0, 1]
    - y_flat (cond_dim,) normalised sensor observations
    - u_flat (true_hw**2,) normalised parameter field
    Returns v_flat same shape as u_flat.

FlatDarcyUNet : eqx.Module
    Wrapper around DarcyUNet satisfying the triangular_transport NNTrainer
    flat-vector interface:
        __call__(xt) -> out
    where xt = [t, y_flat, u_flat] (length 1 + y_dim + u_dim) and
    out = [zeros_y, v_u] (length y_dim + u_dim).
    The y portion of the velocity is always zero (triangular structure).

Topology (default true_hw=33, padded to 36, base_ch=48):
    in_ch = 1 + cond_embed_dim(32) + 2*t_embed_dim(16) = 49
    conv_in  49 → 48,  36×36
    e1       48 → 48,  36×36    [skip1]
    down1    48 → 48,  18×18
    e2       48 → 96,  18×18    [skip2]
    down2    96 → 96,   9×9
    mid1,2   96 → 96,   9×9
    up2      96 → 48,  18×18
    d2      (48+96) → 48, 18×18
    up1      48 → 48,  36×36
    d1      (48+48) → 48, 36×36
    conv_out 48 → 1,   36×36  → crop to 33×33 → flatten (1089,)

~870k parameters at base_ch=48; ~1.6M at base_ch=64.
"""

import equinox as eqx
import jax
import jax.numpy as jnp


# ------------------------------------------------------------------ helpers --

def _fourier_time(t, n_freq):
    """Fourier time features [sin(pi*t), …, sin(n*pi*t), cos(…)] length 2*n_freq."""
    t = jnp.asarray(t).reshape(())
    freqs = jnp.arange(1, n_freq + 1) * jnp.pi
    return jnp.concatenate([jnp.sin(freqs * t), jnp.cos(freqs * t)])


def _bcast(vec, hw):
    """Broadcast 1-D vector of length C to shape (C, hw, hw)."""
    return vec[:, None, None] * jnp.ones((1, hw, hw))


# ---------------------------------------------------------- building blocks --

class _TResBlock(eqx.Module):
    """3×3 conv residual block with FiLM-style per-channel time bias.

    Pattern: GELU(conv1(GELU(x))) + time_proj(t_emb) → GELU → conv2 → + skip.
    A 1×1 skip conv allows in_ch != out_ch.
    """
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d
    tp: eqx.nn.Linear   # time projection
    sk: eqx.nn.Conv2d   # 1×1 skip

    def __init__(self, in_ch, out_ch, t_dim, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.c1 = eqx.nn.Conv2d(in_ch, out_ch, 3, padding=1, key=k1)
        self.c2 = eqx.nn.Conv2d(out_ch, out_ch, 3, padding=1, key=k2)
        self.tp = eqx.nn.Linear(t_dim, out_ch, key=k3)
        self.sk = eqx.nn.Conv2d(in_ch, out_ch, 1, key=k4)

    def __call__(self, x, t_emb):
        h = jax.nn.gelu(self.c1(jax.nn.gelu(x)))
        h = h + self.tp(t_emb)[:, None, None]
        return self.c2(jax.nn.gelu(h)) + self.sk(x)


class _Dn(eqx.Module):
    """Stride-2 3×3 conv — halves both spatial dimensions."""
    conv: eqx.nn.Conv2d

    def __init__(self, in_ch, out_ch, *, key):
        self.conv = eqx.nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, key=key)

    def __call__(self, x):
        return self.conv(x)


class _Up(eqx.Module):
    """Nearest-neighbour 2× upsample + 3×3 conv (avoids checkerboard artifacts)."""
    conv: eqx.nn.Conv2d

    def __init__(self, in_ch, out_ch, *, key):
        self.conv = eqx.nn.Conv2d(in_ch, out_ch, 3, padding=1, key=key)

    def __call__(self, x):
        x = jnp.repeat(jnp.repeat(x, 2, axis=-2), 2, axis=-1)
        return self.conv(x)


# ----------------------------------------------------------------- DarcyUNet -

class DarcyUNet(eqx.Module):
    """Two-level UNet velocity field for the 33×33 Darcy-flow parameter field.

    The field is zero-padded from true_hw (33) to pad_hw (36, next multiple of 4)
    before the UNet so two stride-2 downs give integer spatial sizes:
        36 → 18 → 9.
    The output is cropped back to (true_hw, true_hw) and flattened.

    Parameters
    ----------
    true_hw        : int   spatial side length of the field (33 for Poisson)
    cond_dim       : int   sensor observation dimension (100)
    base_ch        : int   channel width at full resolution (48)
    cond_embed_dim : int   output dimension of the conditioning MLP (32)
    enc_hidden     : int   hidden width of the conditioning MLP (128)
    t_embed_dim    : int   number of Fourier frequency pairs for time (8)
    t_mlp_hidden   : int   hidden width of the time MLP (96)
    key            : PRNGKey
    """

    t_fc1: eqx.nn.Linear
    t_fc2: eqx.nn.Linear
    enc1: eqx.nn.Linear
    enc2: eqx.nn.Linear
    conv_in: eqx.nn.Conv2d
    e1: _TResBlock
    dn1: _Dn
    e2: _TResBlock
    dn2: _Dn
    m1: _TResBlock
    m2: _TResBlock
    up2: _Up
    d2: _TResBlock
    up1: _Up
    d1: _TResBlock
    conv_out: eqx.nn.Conv2d

    true_hw: int = eqx.field(static=True)
    pad_hw: int  = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    cond_embed_dim: int = eqx.field(static=True)
    t_embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        true_hw: int = 33,
        cond_dim: int = 100,
        base_ch: int = 48,
        cond_embed_dim: int = 32,
        enc_hidden: int = 128,
        t_embed_dim: int = 8,
        t_mlp_hidden: int = 96,
        *,
        key,
    ):
        rem    = true_hw % 4
        pad_hw = true_hw if rem == 0 else true_hw + (4 - rem)

        self.true_hw       = true_hw
        self.pad_hw        = pad_hw
        self.cond_dim      = cond_dim
        self.cond_embed_dim = cond_embed_dim
        self.t_embed_dim   = t_embed_dim

        ks  = jax.random.split(key, 16)
        ch2 = 2 * base_ch

        # Time MLP: Fourier features → t_mlp_hidden
        self.t_fc1 = eqx.nn.Linear(2 * t_embed_dim, t_mlp_hidden, key=ks[0])
        self.t_fc2 = eqx.nn.Linear(t_mlp_hidden, t_mlp_hidden, key=ks[1])

        # Conditioning MLP: cond_dim → cond_embed_dim
        self.enc1 = eqx.nn.Linear(cond_dim, enc_hidden, key=ks[2])
        self.enc2 = eqx.nn.Linear(enc_hidden, cond_embed_dim, key=ks[3])

        # UNet
        in_ch        = 1 + cond_embed_dim + 2 * t_embed_dim
        self.conv_in = eqx.nn.Conv2d(in_ch, base_ch, 3, padding=1, key=ks[4])
        self.e1      = _TResBlock(base_ch, base_ch, t_mlp_hidden, key=ks[5])
        self.dn1     = _Dn(base_ch, base_ch, key=ks[6])
        self.e2      = _TResBlock(base_ch, ch2, t_mlp_hidden, key=ks[7])
        self.dn2     = _Dn(ch2, ch2, key=ks[8])
        self.m1      = _TResBlock(ch2, ch2, t_mlp_hidden, key=ks[9])
        self.m2      = _TResBlock(ch2, ch2, t_mlp_hidden, key=ks[10])
        self.up2     = _Up(ch2, base_ch, key=ks[11])
        self.d2      = _TResBlock(base_ch + ch2, base_ch, t_mlp_hidden, key=ks[12])
        self.up1     = _Up(base_ch, base_ch, key=ks[13])
        self.d1      = _TResBlock(base_ch + base_ch, base_ch, t_mlp_hidden, key=ks[14])
        self.conv_out = eqx.nn.Conv2d(base_ch, 1, 3, padding=1, key=ks[15])

    def _temb(self, t):
        return self.t_fc2(jax.nn.silu(self.t_fc1(_fourier_time(t, self.t_embed_dim))))

    def _cemb(self, y):
        return self.enc2(jax.nn.gelu(self.enc1(y)))

    def __call__(self, t, y_flat, u_flat):
        t_s   = jnp.asarray(t).ravel()[0]
        t_emb = self._temb(t_s)
        c_emb = self._cemb(y_flat)

        # Reshape and pad u_flat → (1, pad_hw, pad_hw)
        pad   = self.pad_hw - self.true_hw
        u_pad = jnp.pad(u_flat.reshape(1, self.true_hw, self.true_hw),
                        ((0, 0), (0, pad), (0, pad)))

        h = jnp.concatenate([
            u_pad,
            _bcast(c_emb, self.pad_hw),
            _bcast(_fourier_time(t_s, self.t_embed_dim), self.pad_hw),
        ], axis=0)
        h = self.conv_in(h)

        s1 = self.e1(h, t_emb)
        h  = self.dn1(s1)
        s2 = self.e2(h, t_emb)
        h  = self.dn2(s2)

        h = self.m1(h, t_emb)
        h = self.m2(h, t_emb)

        h = self.up2(h)
        h = self.d2(jnp.concatenate([h, s2], axis=0), t_emb)
        h = self.up1(h)
        h = self.d1(jnp.concatenate([h, s1], axis=0), t_emb)

        out = self.conv_out(h)                                    # (1, pad_hw, pad_hw)
        return out[0, :self.true_hw, :self.true_hw].reshape(-1)   # (true_hw**2,)


# --------------------------------------------------------- FlatDarcyUNet ----

class FlatDarcyUNet(eqx.Module):
    """NNTrainer-compatible flat-vector wrapper around DarcyUNet.

    NNTrainer vmaps the model over a batch and calls it as:
        model(xt)   where  xt = [t, y_flat, u_flat]   shape (1 + y_dim + u_dim,)

    This wrapper splits xt, calls DarcyUNet, and returns:
        [zeros_y, v_u]                                 shape (y_dim + u_dim,)

    The y portion is always zero — the conditioning dimensions do not move
    in the triangular-transport flow.

    Parameters
    ----------
    y_dim   : int   observation dimension (100 for Poisson)
    u_dim   : int   flat field dimension  (1089 = 33² for Poisson)
    true_hw : int   spatial side length (33)
    key     : PRNGKey
    **kwargs        forwarded to DarcyUNet (base_ch, cond_embed_dim, …)
    """
    unet: DarcyUNet
    y_dim: int = eqx.field(static=True)
    u_dim: int = eqx.field(static=True)

    def __init__(self, y_dim: int, u_dim: int, true_hw: int = 33, *, key, **kwargs):
        self.unet = DarcyUNet(true_hw=true_hw, cond_dim=y_dim, **kwargs, key=key)
        self.y_dim = y_dim
        self.u_dim = u_dim

    def __call__(self, xt):
        """xt : (1 + y_dim + u_dim,)  ->  (y_dim + u_dim,)."""
        t   = xt[0]
        y   = xt[1 : 1 + self.y_dim]
        u   = xt[1 + self.y_dim :]
        v_u = self.unet(t, y, u)
        return jnp.concatenate([jnp.zeros(self.y_dim, dtype=v_u.dtype), v_u])
