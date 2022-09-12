import math
class Value:
    """
    Core data structure of Micrograd. Stores a single scalar (data) and its gradient.
    """

    def __init__(self, data, _children = (), _op = '', label = ''):
        """

        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # Operations
    def __add__(self, other): # self + other
        """
        Addition operator
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other): # self * other
        """
        Multiplication operator
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out

    def __neg__(self): # -self
        """
        Negation operator
        """
        return self * -1

    def __sub__(self, other): # self - other
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __pow__(self, other): # self**k
        """
        Power operator
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self, ), f'**{other}')
        
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other): # self / other
        """
        Divison operator
        """
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1
    
    def exp(self): # e**self
        """
        Exponential operator
        """
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out

    def tanh(self):
        """ 
        tanh operation
        Activation function
        """
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        """ 

        """
        return f"Value(data={self.data})"

    

    
