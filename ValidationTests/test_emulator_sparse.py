import sys
sys.path.append('../Emulator')
import emulator
import numpy as np
import scipy.sparse as sp


#==============================================================================#
#                            Test Normalize                                    #
#==============================================================================#
def test_normalize_real():
    # Real already normalized.
    x = sp.dok_matrix((2**100,1), dtype=complex)
    x[100] = 1 + 0j
    assert str(emulator.normalize_sparse(x)) == str(x)
    assert emulator.normalize_sparse(x).shape == x.shape

    # Real needs normalization.
    x = sp.dok_matrix((2**100,1), dtype=complex)
    x[100] = 1 + 0j
    x[1000] = 1 + 0j
    y = sp.dok_matrix((2**100,1), dtype=complex)
    y[100] = 1/np.sqrt(2) + 0j
    y[1000] = 1/np.sqrt(2) + 0j
    assert str(emulator.normalize_sparse(x)) == str(y)
    assert emulator.normalize_sparse(x).shape == y.shape

    # Real positive and negative.
    x = sp.dok_matrix((2**100,1), dtype=complex)
    x[100] = 1 + 0j
    x[1000] = -1 + 0j
    y = sp.dok_matrix((2**100,1), dtype=complex)
    y[100] = 1/np.sqrt(2) + 0j
    y[1000] = -1/np.sqrt(2) + 0j
    assert str(emulator.normalize_sparse(x)) == str(y)
    assert emulator.normalize_sparse(x).shape == y.shape

    # Real entirely Negative.
    x = sp.dok_matrix((2**100,1), dtype=complex)
    x[100] = -1 + 0j
    x[1000] = -1 + 0j
    y = sp.dok_matrix((2**100,1), dtype=complex)
    y[100] = -1/np.sqrt(2) + 0j
    y[1000] = -1/np.sqrt(2) + 0j
    assert str(emulator.normalize_sparse(x)) == str(y)
    assert emulator.normalize_sparse(x).shape == y.shape


def test_normalize_complex():
    # Complex already normal.
    x = sp.dok_matrix((2**100,1), dtype=complex)
    x[100] = 0 + 1j
    assert str(emulator.normalize_sparse(x)) == str(x)
    assert emulator.normalize_sparse(x).shape == x.shape

    # Complex needs normalization.
    x = sp.dok_matrix((2**100,1), dtype=complex)
    x[100] = 0 + 1j
    x[1000] = 0 + 1j
    y = sp.dok_matrix((2**100,1), dtype=complex)
    y[100] = 0 + 1/np.sqrt(2)*1j
    y[1000] = 0 + 1/np.sqrt(2)*1j
    assert str(emulator.normalize_sparse(x)) == str(y)
    assert emulator.normalize_sparse(x).shape == y.shape

    # Complex positive and negative.
    x = sp.dok_matrix((2**100,1), dtype=complex)
    x[100] = 0 + 1j
    x[1000] = 0 - 1j
    y = sp.dok_matrix((2**100,1), dtype=complex)
    y[100] = 0 + 1/np.sqrt(2)*1j
    y[1000] = 0 - 1/np.sqrt(2)*1j
    assert str(emulator.normalize_sparse(x)) == str(y)
    assert emulator.normalize_sparse(x).shape == y.shape

    # Complex entirely negative.
    x = sp.dok_matrix((2**100,1), dtype=complex)
    x[100] = 0 - 1j
    x[1000] = 0 - 1j
    y = sp.dok_matrix((2**100,1), dtype=complex)
    y[100] = 0 - 1/np.sqrt(2)*1j
    y[1000] = 0 - 1/np.sqrt(2)*1j
    assert str(emulator.normalize_sparse(x)) == str(y)
    assert emulator.normalize_sparse(x).shape == y.shape


def test_normalize_real_complex():
    # Real and complex needs normalization.
    x = sp.dok_matrix((2**100,1), dtype=complex)
    x[100] = 1 + 1j
    x[1000] = 1 + 1j
    y = sp.dok_matrix((2**100,1), dtype=complex)
    y[100] = 1/2 + 1/2*1j
    y[1000] = 1/2 + 1/2*1j
    assert str(emulator.normalize_sparse(x)) == str(y)
    assert emulator.normalize_sparse(x).shape == y.shape

    # Full real, part complex.
    x = sp.dok_matrix((2**100,1), dtype=complex)
    x[100] = 1 + 1j
    x[1000] = 0 + 1j
    y = sp.dok_matrix((2**100,1), dtype=complex)
    y[100] = 1/np.sqrt(3) + 1/np.sqrt(3)*1j
    y[1000] = 0 + 1/np.sqrt(3)*1j
    z = emulator.normalize_sparse(x)
    # We perform a complicated check here to deal with rounding errors.
    y_val = []
    z_val = []
    for val in y.values():
        y_val.append(np.round(val, 14))
    for val in z.values():
        z_val.append(np.round(val, 14))
    assert str(z_val) == str(y_val)
    assert str(z.nonzero()) == str(y.nonzero())
    assert z.shape == y.shape

    # One real, one complex.
    x = sp.dok_matrix((2**100,1), dtype=complex)
    x[100] = 1 + 0j
    x[1000] = 0 + 1j
    y = sp.dok_matrix((2**100,1), dtype=complex)
    y[100] = 1/np.sqrt(2) + 0j
    y[1000] = 0 + 1/np.sqrt(2)*1j
    assert str(emulator.normalize_sparse(x)) == str(y)
    assert emulator.normalize_sparse(x).shape == y.shape

    # Positive real, negative complex.
    x = sp.dok_matrix((2**100,1), dtype=complex)
    x[100] = 1 + 0j
    x[1000] = 0 - 1j
    y = sp.dok_matrix((2**100,1), dtype=complex)
    y[100] = 1/np.sqrt(2) + 0j
    y[1000] = 0 - 1/np.sqrt(2)*1j
    assert str(emulator.normalize_sparse(x)) == str(y)
    assert emulator.normalize_sparse(x).shape == y.shape

    # Negative real, positive complex.
    x = sp.dok_matrix((2**100,1), dtype=complex)
    x[100] = -1 + 0j
    x[1000] = 0 + 1j
    y = sp.dok_matrix((2**100,1), dtype=complex)
    y[100] = -1/np.sqrt(2) + 0j
    y[1000] = 0 + 1/np.sqrt(2)*1j
    assert str(emulator.normalize_sparse(x)) == str(y)
    assert emulator.normalize_sparse(x).shape == y.shape


#==============================================================================#
#                               Test Add                                       #
#==============================================================================#
def test_add():
    # Real both positive.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = 1/np.sqrt(2)
    a[1000] = 1/np.sqrt(2)
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = 1
    b[1000] = 1
    c = sp.dok_matrix((2**101,1), dtype=complex)
    c[200] = 1/np.sqrt(2)
    c[1100] = 2/np.sqrt(2)
    c[2000] = 1/np.sqrt(2)
    assert str(emulator.add_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.add_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Real both negative.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = -1/np.sqrt(2)
    a[1000] = -1/np.sqrt(2)
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = -1
    b[1000] = -1
    c = sp.dok_matrix((2**101,1), dtype=complex)
    c[200] = 1/np.sqrt(2)
    c[1100] = 2/np.sqrt(2)
    c[2000] = 1/np.sqrt(2)
    assert str(emulator.add_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.add_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Real positive and negative.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = 1/np.sqrt(2)
    a[1000] = 1/np.sqrt(2)
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = -1
    b[1000] = 1
    c = sp.dok_matrix((2**101,1), dtype=complex)
    c[200] = -1/np.sqrt(2)
    c[2000] = 1/np.sqrt(2)
    assert str(emulator.add_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.add_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Complex both positive.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = 1/np.sqrt(2)*1j
    a[1000] = 1/np.sqrt(2)*1j
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = 1j
    b[1000] = 1j
    c = sp.dok_matrix((2**101,1), dtype=complex)
    c[200] = -1/np.sqrt(2)
    c[1100] = -2/np.sqrt(2)
    c[2000] = -1/np.sqrt(2)
    assert str(emulator.add_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.add_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Complex both negative.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = -1/np.sqrt(2)*1j
    a[1000] = -1/np.sqrt(2)*1j
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = -1j
    b[1000] = -1j
    c = sp.dok_matrix((2**101,1), dtype=complex)
    c[200] = -1/np.sqrt(2)
    c[1100] = -2/np.sqrt(2)
    c[2000] = -1/np.sqrt(2)
    assert str(emulator.add_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.add_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Complex positive and negative.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = 1/np.sqrt(2)*1j
    a[1000] = 1/np.sqrt(2)*1j
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = -1j
    b[1000] = 1j
    c = sp.dok_matrix((2**101,1), dtype=complex)
    c[200] = 1/np.sqrt(2)
    c[2000] = -1/np.sqrt(2)
    assert str(emulator.add_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.add_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Real and complex.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = 1/np.sqrt(2) + 1/np.sqrt(2)*1j
    a[1000] = 1/np.sqrt(2) - 1/np.sqrt(2)*1j
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = 1 + 0j
    b[1000] = 0 + 1j
    c = sp.dok_matrix((2**101,1), dtype=complex)
    c[200] = 1/np.sqrt(2) + 1/np.sqrt(2)*1j
    c[2000] = 1/np.sqrt(2) + 1/np.sqrt(2)*1j
    assert str(emulator.add_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.add_sparse(a, b).shape == emulator.normalize_sparse(c).shape


#==============================================================================#
#                              Test Multiply                                   #
#==============================================================================#
def test_mult():
    # Real both positive.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = 1/np.sqrt(2)
    a[1000] = 1/np.sqrt(2)
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = 1
    b[1000] = 1
    c = sp.dok_matrix(((2**100)*(2**100),1), dtype=complex)
    c[10000] = 1/np.sqrt(2)
    c[100000] = 2/np.sqrt(2)
    c[1000000] = 1/np.sqrt(2)
    assert str(emulator.multiply_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.multiply_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Real both negative.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = -1/np.sqrt(2)
    a[1000] = -1/np.sqrt(2)
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = -1
    b[1000] = -1
    c = sp.dok_matrix(((2**100)*(2**100),1), dtype=complex)
    c[10000] = 1/np.sqrt(2)
    c[100000] = 2/np.sqrt(2)
    c[1000000] = 1/np.sqrt(2)
    assert str(emulator.multiply_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.multiply_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Real positive and negative.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = 1/np.sqrt(2)
    a[1000] = 1/np.sqrt(2)
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = -1
    b[1000] = 1
    c = sp.dok_matrix(((2**100)*(2**100),1), dtype=complex)
    c[10000] = -1/np.sqrt(2)
    c[1000000] = 1/np.sqrt(2)
    assert str(emulator.multiply_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.multiply_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Complex both positive.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = 1/np.sqrt(2)*1j
    a[1000] = 1/np.sqrt(2)*1j
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = 1j
    b[1000] = 1j
    c = sp.dok_matrix(((2**100)*(2**100),1), dtype=complex)
    c[10000] = -1/np.sqrt(2)
    c[100000] = -2/np.sqrt(2)
    c[1000000] = -1/np.sqrt(2)
    assert str(emulator.multiply_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.multiply_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Complex both negative.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = -1/np.sqrt(2)*1j
    a[1000] = -1/np.sqrt(2)*1j
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = -1j
    b[1000] = -1j
    c = sp.dok_matrix(((2**100)*(2**100),1), dtype=complex)
    c[10000] = -1/np.sqrt(2)
    c[100000] = -2/np.sqrt(2)
    c[1000000] = -1/np.sqrt(2)
    assert str(emulator.multiply_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.multiply_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Complex positive and negative.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = 1/np.sqrt(2)*1j
    a[1000] = 1/np.sqrt(2)*1j
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = -1j
    b[1000] = 1j
    c = sp.dok_matrix(((2**100)*(2**100),1), dtype=complex)
    c[10000] = 1/np.sqrt(2)
    c[1000000] = -1/np.sqrt(2)
    assert str(emulator.multiply_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.multiply_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Real and complex.
    a = sp.dok_matrix((2**100,1), dtype=complex)
    a[100] = 1/np.sqrt(2) + 1/np.sqrt(2)*1j
    a[1000] = 1/np.sqrt(2) - 1/np.sqrt(2)*1j
    b = sp.dok_matrix((2**100,1), dtype=complex)
    b[100] = 1 + 0j
    b[1000] = 0 + 1j
    c = sp.dok_matrix(((2**100)*(2**100),1), dtype=complex)
    c[10000] = 1/np.sqrt(2) + 1/np.sqrt(2)*1j
    c[1000000] = 1/np.sqrt(2) + 1/np.sqrt(2)*1j
    assert str(emulator.multiply_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.multiply_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    

#==============================================================================#
#                            Test Exponentiate                                 #
#==============================================================================#
def test_exp():
    # Real both positive.
    a = sp.dok_matrix((2**10,1), dtype=complex)
    a[10] = 1/np.sqrt(2)
    a[100] = 1/np.sqrt(2)
    b = sp.dok_matrix((2**2,1), dtype=complex)
    b[2] = 1
    b[3] = 1
    c = sp.dok_matrix(((2**10)**(2**2),1), dtype=complex)
    c[10**2] = 1/np.sqrt(2)
    c[10**3] = 1/np.sqrt(2)
    c[100**2] = 1/np.sqrt(2)
    c[100**3] = 1/np.sqrt(2)
    assert str(emulator.exponentiate_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.exponentiate_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Real both negative.
    a = sp.dok_matrix((2**10,1), dtype=complex)
    a[10] = -1/np.sqrt(2)
    a[100] = -1/np.sqrt(2)
    b = sp.dok_matrix((2**2,1), dtype=complex)
    b[2] = -1
    b[3] = -1
    c = sp.dok_matrix(((2**10)**(2**2),1), dtype=complex)
    c[10**2] = 1/np.sqrt(2)
    c[10**3] = 1/np.sqrt(2)
    c[100**2] = 1/np.sqrt(2)
    c[100**3] = 1/np.sqrt(2)
    assert str(emulator.exponentiate_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.exponentiate_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Real positive and negative.
    a = sp.dok_matrix((2**10,1), dtype=complex)
    a[10] = 1/np.sqrt(2)
    a[100] = 1/np.sqrt(2)
    b = sp.dok_matrix((2**2,1), dtype=complex)
    b[2] = -1
    b[3] = 1
    c = sp.dok_matrix(((2**10)**(2**2),1), dtype=complex)
    c[10**2] = -1/np.sqrt(2)
    c[10**3] = 1/np.sqrt(2)
    c[100**2] = -1/np.sqrt(2)
    c[100**3] = 1/np.sqrt(2)
    assert str(emulator.exponentiate_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.exponentiate_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Complex both positive.
    a = sp.dok_matrix((2**10,1), dtype=complex)
    a[10] = 1/np.sqrt(2)*1j
    a[100] = 1/np.sqrt(2)*1j
    b = sp.dok_matrix((2**2,1), dtype=complex)
    b[2] = 1j
    b[3] = 1j
    c = sp.dok_matrix(((2**10)**(2**2),1), dtype=complex)
    c[10**2] = -1/np.sqrt(2)
    c[10**3] = -1/np.sqrt(2)
    c[100**2] = -1/np.sqrt(2)
    c[100**3] = -1/np.sqrt(2)
    assert str(emulator.exponentiate_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.exponentiate_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Complex both negative.
    a = sp.dok_matrix((2**10,1), dtype=complex)
    a[10] = -1/np.sqrt(2)*1j
    a[100] = -1/np.sqrt(2)*1j
    b = sp.dok_matrix((2**2,1), dtype=complex)
    b[2] = -1j
    b[3] = -1j
    c = sp.dok_matrix(((2**10)**(2**2),1), dtype=complex)
    c[10**2] = -1/np.sqrt(2)
    c[10**3] = -1/np.sqrt(2)
    c[100**2] = -1/np.sqrt(2)
    c[100**3] = -1/np.sqrt(2)
    assert str(emulator.exponentiate_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.exponentiate_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Complex positive and negative.
    a = sp.dok_matrix((2**10,1), dtype=complex)
    a[10] = 1/np.sqrt(2)*1j
    a[100] = 1/np.sqrt(2)*1j
    b = sp.dok_matrix((2**2,1), dtype=complex)
    b[2] = -1j
    b[3] = 1j
    c = sp.dok_matrix(((2**10)**(2**2),1), dtype=complex)
    c[10**2] = 1/np.sqrt(2)
    c[10**3] = -1/np.sqrt(2)
    c[100**2] = 1/np.sqrt(2)
    c[100**3] = -1/np.sqrt(2)
    assert str(emulator.exponentiate_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.exponentiate_sparse(a, b).shape == emulator.normalize_sparse(c).shape
    
    # Real and complex.
    a = sp.dok_matrix((2**10,1), dtype=complex)
    a[10] = 1/np.sqrt(2) + 1/np.sqrt(2)*1j
    a[100] = 1/np.sqrt(2) - 1/np.sqrt(2)*1j
    b = sp.dok_matrix((2**2,1), dtype=complex)
    b[2] = 1 + 0j
    b[3] = 0 + 1j
    c = sp.dok_matrix(((2**10)**(2**2),1), dtype=complex)
    c[10**2] = 1/np.sqrt(2) + 1/np.sqrt(2)*1j
    c[10**3] = -1/np.sqrt(2) + 1/np.sqrt(2)*1j
    c[100**2] = 1/np.sqrt(2) - 1/np.sqrt(2)*1j
    c[100**3] = 1/np.sqrt(2) + 1/np.sqrt(2)*1j
    assert str(emulator.exponentiate_sparse(a, b)) == str(emulator.normalize_sparse(c))
    assert emulator.exponentiate_sparse(a, b).shape == emulator.normalize_sparse(c).shape