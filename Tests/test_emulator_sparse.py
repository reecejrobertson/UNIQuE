import sys
sys.path.append('../../Emulator')
import emulator
import numpy as np


#==============================================================================#
#                            Test Normalize                                    #
#==============================================================================#
def test_normalize_real():
    # Real already normalized.
    x = np.array([1, 0], dtype=complex)
    assert np.allclose(emulator.normalize(x), x)

    # Real needs normalization.
    x = np.array([1, 1], dtype=complex)
    y = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    assert np.allclose(emulator.normalize(x), y)

    # Real positive and negative.
    x = np.array([1, -1], dtype=complex)
    y = np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
    assert np.allclose(emulator.normalize(x), y)

    # Real entirely Negative.
    x = np.array([-1, -1], dtype=complex)
    y = np.array([-1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
    assert np.allclose(emulator.normalize(x), y)


def test_normalize_complex():
    # Complex already normal.
    x = np.array([0+1j, 0+0j], dtype=complex)
    assert np.allclose(emulator.normalize(x), x)

    # Complex needs normalization.
    x = np.array([0+1j, 0+1j], dtype=complex)
    y = np.array([0+1/np.sqrt(2)*1j, 0+1/np.sqrt(2)*1j], dtype=complex)
    assert np.allclose(emulator.normalize(x), y)

    # Complex positive and negative.
    x = np.array([0+1j, 0-1j], dtype=complex)
    y = np.array([0+1/np.sqrt(2)*1j, 0-1/np.sqrt(2)*1j], dtype=complex)
    assert np.allclose(emulator.normalize(x), y)

    # Complex entirely negative.
    x = np.array([0-1j, 0-1j], dtype=complex)
    y = np.array([0-1/np.sqrt(2)*1j, 0-1/np.sqrt(2)*1j], dtype=complex)
    assert np.allclose(emulator.normalize(x), y)


def test_normalize_real_complex():
    # Real and complex needs normalization.
    x = np.array([1+1j, 1+1j], dtype=complex)
    y = np.array([1/2+1/2*1j, 1/2+1/2*1j], dtype=complex)
    assert np.allclose(emulator.normalize(x), y)

    # Full real, part complex.
    x = np.array([1+1j, 0+1j], dtype=complex)
    y = np.array([1/np.sqrt(3)+1/np.sqrt(3)*1j, 0+1/np.sqrt(3)*1j], dtype=complex)
    assert np.allclose(emulator.normalize(x), y)

    # One real, one complex.
    x = np.array([1+0j, 0+1j], dtype=complex)
    y = np.array([1/np.sqrt(2)+0j, 0+1/np.sqrt(2)*1j], dtype=complex)
    assert np.allclose(emulator.normalize(x), y)

    # Positive real, negative complex.
    x = np.array([1+0j, 0-1j], dtype=complex)
    y = np.array([1/np.sqrt(2)+0j, 0-1/np.sqrt(2)*1j], dtype=complex)
    assert np.allclose(emulator.normalize(x), y)

    # Negative real, positive complex.
    x = np.array([-1+0j, 0+1j], dtype=complex)
    y = np.array([-1/np.sqrt(2)+0j, 0+1/np.sqrt(2)*1j], dtype=complex)
    assert np.allclose(emulator.normalize(x), y)


#==============================================================================#
#                               Test Add                                       #
#==============================================================================#
def test_add():
    # Real both positive.
    a = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    b = np.array([1, 1], dtype=complex)
    c = np.array([1/np.sqrt(2), 2/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)
    assert(np.allclose(emulator.add(a, b), emulator.normalize(c)))
    
    # Real both negative.
    a = np.array([-1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
    b = np.array([-1, -1], dtype=complex)
    c = np.array([1/np.sqrt(2), 2/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)
    assert(np.allclose(emulator.add(a, b), emulator.normalize(c)))
    
    # Real positive and negative.
    a = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    b = np.array([-1, 1], dtype=complex)
    c = np.array([-1/np.sqrt(2), 0, 1/np.sqrt(2), 0], dtype=complex)
    assert(np.allclose(emulator.add(a, b), emulator.normalize(c)))
    
    # Complex both positive.
    a = np.array([1/np.sqrt(2)*1j, 1/np.sqrt(2)*1j], dtype=complex)
    b = np.array([1j, 1j], dtype=complex)
    c = np.array([-1/np.sqrt(2), -2/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
    assert(np.allclose(emulator.add(a, b), emulator.normalize(c)))
    
    # Complex both negative.
    a = np.array([-1/np.sqrt(2)*1j, -1/np.sqrt(2)*1j], dtype=complex)
    b = np.array([-1j, -1j], dtype=complex)
    c = np.array([-1/np.sqrt(2), -2/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
    assert(np.allclose(emulator.add(a, b), emulator.normalize(c)))
    
    # Complex positive and negative.
    a = np.array([1/np.sqrt(2)*1j, 1/np.sqrt(2)*1j], dtype=complex)
    b = np.array([-1j, 1j], dtype=complex)
    c = np.array([1/np.sqrt(2), 0, -1/np.sqrt(2), 0], dtype=complex)
    assert(np.allclose(emulator.add(a, b), emulator.normalize(c)))
    
    # Real and complex.
    a = np.array([1/2+1/2*1j, 1/2-1/2*1j], dtype=complex)
    b = np.array([1+0j, 0+1j], dtype=complex)
    c = np.array([1/2+1/2*1j, 0, 1/2+1/2*1j, 0], dtype=complex)
    assert(np.allclose(emulator.add(a, b), emulator.normalize(c)))


#==============================================================================#
#                              Test Multiply                                   #
#==============================================================================#
def test_mult():
    # Real both positive.
    a = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    b = np.array([1, 1], dtype=complex)
    c = np.array([3/np.sqrt(2), 1/np.sqrt(2), 0, 0], dtype=complex)
    assert(np.allclose(emulator.multiply(a, b), emulator.normalize(c)))
    
    # Real both negative.
    a = np.array([-1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
    b = np.array([-1, -1], dtype=complex)
    c = np.array([3/np.sqrt(2), 1/np.sqrt(2), 0, 0], dtype=complex)
    assert(np.allclose(emulator.multiply(a, b), emulator.normalize(c)))
    
    # Real positive and negative.
    a = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    b = np.array([-1, 1], dtype=complex)
    c = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0, 0], dtype=complex)
    assert(np.allclose(emulator.multiply(a, b), emulator.normalize(c)))
    
    # Complex both positive.
    a = np.array([1/np.sqrt(2)*1j, 1/np.sqrt(2)*1j], dtype=complex)
    b = np.array([1j, 1j], dtype=complex)
    c = np.array([-3/np.sqrt(2), -1/np.sqrt(2), 0, 0], dtype=complex)
    assert(np.allclose(emulator.multiply(a, b), emulator.normalize(c)))
    
    # Complex both negative.
    a = np.array([-1/np.sqrt(2)*1j, -1/np.sqrt(2)*1j], dtype=complex)
    b = np.array([-1j, -1j], dtype=complex)
    c = np.array([-3/np.sqrt(2), -1/np.sqrt(2), 0, 0], dtype=complex)
    assert(np.allclose(emulator.multiply(a, b), emulator.normalize(c)))
    
    # Complex positive and negative.
    a = np.array([1/np.sqrt(2)*1j, 1/np.sqrt(2)*1j], dtype=complex)
    b = np.array([-1j, 1j], dtype=complex)
    c = np.array([1/np.sqrt(2), -1/np.sqrt(2), 0, 0], dtype=complex)
    assert(np.allclose(emulator.multiply(a, b), emulator.normalize(c)))
    
    # Real and complex.
    a = np.array([1/2+1/2*1j, 1/2-1/2*1j], dtype=complex)
    b = np.array([1+0j, 0+1j], dtype=complex)
    c = np.array([1/2+1/2*1j, 1/2+1/2*1j, 0, 0], dtype=complex)
    assert(np.allclose(emulator.multiply(a, b), emulator.normalize(c)))
    

#==============================================================================#
#                            Test Exponentiate                                 #
#==============================================================================#
def test_exp():
    # Real both positive.
    a = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    b = np.array([1, 1], dtype=complex)
    c = np.array([1/np.sqrt(2), 3/np.sqrt(2), 0, 0], dtype=complex)
    assert(np.allclose(emulator.exponentiate(a, b), emulator.normalize(c)))
    
    # Real both negative.
    a = np.array([-1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
    b = np.array([-1, -1], dtype=complex)
    c = np.array([1/np.sqrt(2), 3/np.sqrt(2), 0, 0], dtype=complex)
    assert(np.allclose(emulator.exponentiate(a, b), emulator.normalize(c)))
    
    # Real positive and negative.
    a = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    b = np.array([-1, 1], dtype=complex)
    c = np.array([1/np.sqrt(2), -1/np.sqrt(2), 0, 0], dtype=complex)
    assert(np.allclose(emulator.exponentiate(a, b), emulator.normalize(c)))
    
    # Complex both positive.
    a = np.array([1/np.sqrt(2)*1j, 1/np.sqrt(2)*1j], dtype=complex)
    b = np.array([1j, 1j], dtype=complex)
    c = np.array([-1/np.sqrt(2), -3/np.sqrt(2), 0, 0], dtype=complex)
    assert(np.allclose(emulator.exponentiate(a, b), emulator.normalize(c)))
    
    # Complex both negative.
    a = np.array([-1/np.sqrt(2)*1j, -1/np.sqrt(2)*1j], dtype=complex)
    b = np.array([-1j, -1j], dtype=complex)
    c = np.array([-1/np.sqrt(2), -3/np.sqrt(2), 0, 0], dtype=complex)
    assert(np.allclose(emulator.exponentiate(a, b), emulator.normalize(c)))
    
    # Complex positive and negative.
    a = np.array([1/np.sqrt(2)*1j, 1/np.sqrt(2)*1j], dtype=complex)
    b = np.array([-1j, 1j], dtype=complex)
    c = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0, 0], dtype=complex)
    assert(np.allclose(emulator.exponentiate(a, b), emulator.normalize(c)))
    
    # Real and complex.
    a = np.array([1/2+1/2*1j, 1/2-1/2*1j], dtype=complex)
    b = np.array([1+0j, 0+1j], dtype=complex)
    c = np.array([-1/2+1/2*1j, 3/2+1/2*1j, 0, 0], dtype=complex)
    assert(np.allclose(emulator.exponentiate(a, b), emulator.normalize(c)))