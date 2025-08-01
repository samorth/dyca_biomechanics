import numpy as np
import scipy.linalg as linalg


def _derivativesignal(signal: np.ndarray, time_signal: np.ndarray):
    """Compute the derivative of a signal with respect to its sampling rate

    Arguments:
        signal {np.ndarray} -- Signal to be differentiated (timepoints, channels).
        time_signal {np.ndarray} -- Time vector of the signal (timepoints, ).

    Returns:
        np.ndarray -- derivative of the signal (timepoints, channels)
    """
    max = len(signal)
    dt = time_signal[1] - time_signal[0]
    derivative_signal = (signal[2:max, :] - signal[0:max - 2, :]) / (2 * dt)
    first = (signal[1, :] - signal[0, :]) / dt
    last = (signal[max - 1, :] - signal[max - 2, :]) / dt
    derivative_signal = np.vstack((first, derivative_signal, last))

    return derivative_signal


def _input_check(signal: np.ndarray, m: int, n: int, time_index: np.ndarray, derivative_signal: np.ndarray):
    """Check the input of the dyca function"""

    if signal is None:
        raise ValueError('No signal provided')
    if not isinstance(signal, np.ndarray):
        raise ValueError('Signal has to be a numpy array')
    if signal.ndim != 2:
        raise ValueError('Signal has to be a 2D array')
    if signal.shape[0] < signal.shape[1]:
        raise ValueError('Signal has to have more timepoints than channels')

    if not isinstance(time_index, np.ndarray):
        raise ValueError('Time signal has to be a numpy array')
    if time_index.ndim != 1:
        raise ValueError('Time signal has to be a 1D array')
    if time_index.shape[0] != signal.shape[0]:
        raise ValueError('Time signal has to have the same length as the signal')

    if not isinstance(derivative_signal, np.ndarray):
        raise ValueError('Derivative signal has to be a numpy array')
    if derivative_signal.shape != signal.shape:
        raise ValueError('Derivative signal has to have the same shape as the signal')

    if (n is not None):

        if not isinstance(n, int) or n < -1 or n > signal.shape[1] or n == 0:
            raise ValueError('n has to be an integer greater than 0, or -1 for no limit')

    if (m is not None):
        if not isinstance(m, int) or m < -1 or m > signal.shape[1] or m == 0:
            raise ValueError('m has to be an integer greater than 0, or -1 for no limit')

    if (m is not None and n is not None):
        # dyca conditions m >= n - m
        if m < n - m:
            raise ValueError('m has to be greater than or equal to n - m')
        if m > signal.shape[1]:
            raise ValueError('m has to be smaller than the number of channels')
        if n > signal.shape[1]:
            raise ValueError('n has to be smaller than the number of channels')
        if n < m:
            raise ValueError('n has to be greater than or equal to m')


def _cholesky_inverse(matrix: np.ndarray):
    """Calculate the inverse of a matrix by Cholesky"""
    L_lower_matrix = linalg.cholesky(matrix, lower=True)
    L_lower_inv_matrix = linalg.inv(L_lower_matrix)
    matrix_inv = L_lower_inv_matrix.T @ L_lower_inv_matrix
    return matrix_inv


def _check_eigenvalues_real(eigenvalues: np.ndarray):
    """ Check if the eigenvalues are real"""
    if np.all(np.imag(eigenvalues) != 0):
        raise ValueError('Complex eigenvalues detected. Check your input signal.')
    return np.real(eigenvalues)


def _calculate_correlations(signal: np.ndarray, derivative_signal: np.ndarray) -> tuple:
    """Calculate the correlation matrices C0, C1 and C2"""
    time_length = signal.shape[0]

    # Calculate correlation matrices C0, C1 and C2
    C0 = (signal.T @ signal) / time_length
    C1 = (derivative_signal.T @ signal) / time_length
    C2 = (derivative_signal.T @ derivative_signal) / time_length

    # calculate the inverse of C0 by Cholesky decomposition
    try:
        C0_inv = _cholesky_inverse(C0)
    except Exception as e:
        print(e)
        raise ValueError('Failed calculating the inverse of C0. Check your rank of the input signal.')

    return C0_inv, C1, C2


def _calculate_eigenvalues_and_vectors(C0_inv: np.ndarray, C1: np.ndarray, C2: np.ndarray) -> tuple:
    """ Calculate the eigenvalues and eigenvectors of the generalized eigenvalue problem"""
    # Solve the generalized eigenvalue problem
    # C1* inv(C0)* C1^T* u_i = lambda_i * C2 * u_i
    eigenvalues, eigenvectors = linalg.eig(C1 @ C0_inv @ C1.T, C2)

    # Check if eigenvalues are real
    eigenvalues = _check_eigenvalues_real(eigenvalues)

    # Sort eigenvalues and eigenvectors according to absolute value of eigenvalues
    indices = np.flip(np.argsort(eigenvalues))
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    return eigenvalues, eigenvectors

def _calculate_amplitudes(m: int, C0_inv: np.ndarray, C1: np.ndarray, eigenvectors: np.ndarray, signal: np.ndarray) -> tuple:
    """ Calculate the amplitudes and the modes_V of the DyCA decomposition"""
    # Stop algorithm, if m is unknown
    if m is not None:
        # Select m linear components
        eigenvectors = eigenvectors[:, :m]

        # Calculate associated matrix V
        V = C0_inv @ C1.T @ eigenvectors

        # calculating the svd to select n equations
        projectionMatrix = np.concatenate((eigenvectors, V), axis=1)
        amplitude = signal @ projectionMatrix
        amplitude_norm = np.divide(amplitude, np.sqrt(np.diag(amplitude.T @ amplitude))).T
    else:
        V = None
        amplitude_norm = None
    return V, amplitude_norm



def _calculate_svd_amplitudes(amplitudes: np.ndarray, n: int) -> dict:
    """ Calculate the SVD amplitudes of the DyCA decomposition"""
    # We can only calculate the SVD if amplitudes are known
    if amplitudes is not None:
        U_svd, S_svd, V_svd = linalg.svd(amplitudes, full_matrices=False)
        # We can only project if n is known
        if n is not None:
            # Truncation of the SVD to n components yields the amplitudes
            amplitudes_svd = U_svd[0:n, 0:n] @ np.diag(S_svd[0:n]) @ V_svd[0:n, :]
        else:
            amplitudes_svd = None

    return amplitudes_svd, S_svd