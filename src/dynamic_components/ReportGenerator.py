import numpy as np

class ReportGenerator:
    def __init__(self, dyca_output: dict):
        self.amplitudes = dyca_output.get('amplitudes')
        self.eigenvalues = dyca_output.get('generalized_eigenvalues')
        self.modes_U = dyca_output.get('modes_U')
        self.modes_V = dyca_output.get('modes_V')

    def _array_info(self, arr: np.ndarray, name: str) -> str:
        lines = []
        lines.append(f"{name}: shape={arr.shape}, ndim={arr.ndim}, size={arr.size}")

        if np.isnan(arr).any():
            nan_indices = np.argwhere(np.isnan(arr))
            lines.append(f"  Contains NaNs at indices: {nan_indices.tolist()}")
        else:
            lines.append("  No NaNs found.")

        return "\n".join(lines)

    def _complex_info(self, arr: np.ndarray, name: str) -> str:
        lines = []
        if np.iscomplexobj(arr):
            real = np.real(arr)
            imag = np.imag(arr)
            mean_real = np.mean(np.abs(real))
            mean_imag = np.mean(np.abs(imag))
            ratio = mean_imag / mean_real if mean_real != 0 else np.inf
            lines.append(f"{name} is complex-valued.")
            lines.append(f"  Mean(|Imag|)/Mean(|Real|) = {ratio:.4f}")

            mask = np.abs(imag) > 0.05 * np.abs(real)
            if mask.any():
                indices = np.argwhere(mask)
                lines.append(f"  Imaginary part >5% of real part at indices: {indices.tolist()}")
            else:
                lines.append("  No elements with imaginary part >5% of real part.")
        else:
            lines.append(f"{name} is real-valued.")
        return "\n".join(lines)

    def _eigenvalues_info(self) -> str:
        arr = self.eigenvalues
        lines = []
        lines.append(f"eigenvalues: shape={arr.shape}, ndim={arr.ndim}, size={arr.size}")
        
        if np.isnan(arr).any():
            nan_idx = np.argwhere(np.isnan(arr))
            lines.append(f"  Contains NaNs at indices: {nan_idx.tolist()}")
        
        if np.isposinf(arr).any() or np.isneginf(arr).any():
            inf_idx = np.argwhere(np.isinf(arr))
            lines.append(f"  Contains inf at indices: {inf_idx.tolist()}")
        
        if (arr < 0).any():
            neg_idx = np.argwhere(arr < 0)
            lines.append(f"  Contains negative values at indices: {neg_idx.tolist()}")
        if not any([np.isnan(arr).any(), np.isinf(arr).any(), (arr < 0).any()]):
            lines.append("  No NaNs, inf, or negative values found.")
        return "\n".join(lines)

    def generate_report(self) -> str:
        parts = []
        
        parts.append(self._array_info(self.amplitudes, "amplitudes"))
        parts.append(self._array_info(self.eigenvalues, "eigenvalues"))
        parts.append(self._array_info(self.modes_U, "modes_U"))
        parts.append(self._array_info(self.modes_V, "modes_V"))

        
        parts.append(self._complex_info(self.amplitudes, "amplitudes"))
        parts.append(self._complex_info(self.modes_U, "modes_U"))
        parts.append(self._complex_info(self.modes_V, "modes_V"))

        
        parts.append(self._eigenvalues_info())

        return "\n\n".join(parts)


# Example usage:
# report = ReportGenerator(dyca_output)
# print(report.generate_report())
