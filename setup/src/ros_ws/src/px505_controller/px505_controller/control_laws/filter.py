import numpy as np

class DigitalFilter:
    """
    Generic discrete-time IIR/FIR filter with user-specified coefficients.
    
    Implements the difference equation:
    
        y[k] = (1/a0) * ( sum_{i=0..M} b[i] * x[k-i] 
                          - sum_{j=1..N} a[j] * y[k-j] )
    
    where:
        - a[0] is normalized to 1.0 internally
        - x, y can be vectors (one filter per channel)
    """
    b: np.ndarray
    a: np.ndarray
    
    def __init__(self, b: np.ndarray, a: np.ndarray = None, n_channels: int = 1):
        """
        Args:
            b (np.ndarray): Numerator coefficients (shape (M+1,))
            a (np.ndarray): Denominator coefficients (shape (N+1,)), a[0] != 0.
                            If None, a = [1.0] (pure FIR filter).
            n_channels (int): Number of parallel channels (e.g. 3 for x,y,z).
        """
        b = np.asarray(b, dtype=float).flatten()
        if a is None:
            a = np.array([1.0], dtype=float)
        else:
            a = np.asarray(a, dtype=float).flatten()
            
        if a[0] == 0.0:
            raise ValueError("a[0] must be non-zero for a valid filter.")
        
        # Normalize so that a[0] = 1
        self.b = b / a[0]
        self.a = a / a[0]
        
        self.order_b = self.b.shape[0]   # M+1
        self.order_a = self.a.shape[0]   # N+1
        
        self.n_channels = int(n_channels)
        
        # History buffers: shape (n_channels, order_*)
        self.x_hist = np.zeros((self.n_channels, self.order_b))
        self.y_hist = np.zeros((self.n_channels, self.order_a))
        
    def reset(self, value: np.ndarray = None):
        """
        Reset filter state.
        
        Args:
            value (np.ndarray): Optional initial output value per channel.
                                If None, state is reset to zeros.
        """
        self.x_hist[:] = 0.0
        self.y_hist[:] = 0.0
        
        if value is not None:
            value = np.asarray(value, dtype=float).flatten()
            if value.shape[0] != self.n_channels:
                raise ValueError(f"Expected value of shape ({self.n_channels},), got {value.shape}.")
            # Initialize y history to this value
            self.y_hist[:, :] = value.reshape(-1, 1)
    
    def filter(self, x: np.ndarray) -> np.ndarray:
        """
        Filter a new input sample.
        
        Args:
            x (np.ndarray): Current input sample per channel, shape (n_channels,)
        
        Returns:
            np.ndarray: Current output sample per channel, shape (n_channels,)
        """
        x = np.asarray(x, dtype=float).flatten()
        if x.shape[0] != self.n_channels:
            raise ValueError(f"Expected x of shape ({self.n_channels},), got {x.shape}.")
        
        # Shift input history: [x[k-1], x[k-2], ...]
        if self.order_b > 1:
            self.x_hist[:, 1:] = self.x_hist[:, :-1]
        self.x_hist[:, 0] = x
        
        y = np.zeros(self.n_channels, dtype=float)
        
        # Compute output for each channel
        for ch in range(self.n_channels):
            # FIR part: sum b[i] * x[k-i]
            acc = np.dot(self.b, self.x_hist[ch, :])
            
            # IIR part: - sum a[j] * y[k-j], j=1..N
            if self.order_a > 1:
                acc -= np.dot(self.a[1:], self.y_hist[ch, :self.order_a-1])
            
            y[ch] = acc
        
        # Shift output history: [y[k-1], y[k-2], ...]
        if self.order_a > 1:
            self.y_hist[:, 1:] = self.y_hist[:, :-1]
        self.y_hist[:, 0] = y
        
        return y
