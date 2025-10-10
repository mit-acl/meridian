import pytest
import numpy as np
import robotdatapy.transform as rdpt

from gen_seg_match.segment.segment_types import SegmentPoint, SegmentLine, SegmentPlane
from gen_seg_match.match.segment_matcher import SegmentMatcher, SegmentMatcherParams

@pytest.fixture
def three_line_segments():
    
    linea1 = SegmentLine(
        id=1,
        point=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        cos_feature=np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
    )
    
    linea2 = SegmentLine(
        id=2,
        point=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 1.0]),
        cos_feature=np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0]),
    )
    
    linea3 = SegmentLine(
        id=3,
        point=np.array([2.0, 0.0, 1.0]),
        direction=np.array([0.0, 1.0, 0.0]),
        cos_feature=np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]),
    )
    
    lineb1 = linea1.copy()
    lineb2 = linea2.copy()
    lineb3 = linea3.copy()
    
    lineb1.endpoints = (np.array([-1.0, 0.0, 0.0]), None)
    lineb2.endpoints = (np.array([-1.0, 0.0, -1.0]), np.array([2.0, 0.0, 2.0]))
    
    linesa = [linea1, linea2, linea3]
    linesb = [lineb1, lineb2, lineb3]
    
    # for line in linesb:
    #     line.transform(rdpt.xyz_rpy_to_transform(np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, np.pi/6])))
        
    return linesa, linesb

def test_lines_a(three_line_segments):
    linesa, _ = three_line_segments
    assert len(linesa) == 3
    assert all(isinstance(line, SegmentLine) for line in linesa)
    
    assert linesa[0].to_array().shape == (14,)  # type + point(3) + direction(3) + num_endpoints + cos_feature(6)
    assert np.allclose(linesa[0].to_array(), 
                       np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0, 0.0, 0.0,]))
    
    assert linesa[1].to_array().shape == (14,)
    assert np.allclose(linesa[1].to_array(), 
                       np.array([1.0, 0.0, 0.0, 0.0, 1.0/np.sqrt(2), 0.0, 1.0/np.sqrt(2), 0.0, np.sqrt(2)/2, 0.0, np.sqrt(2)/2, 0.0, 0.0, 0.0,]))
    
    assert linesa[2].to_array().shape == (14,)
    assert np.allclose(linesa[2].to_array(), 
                       np.array([1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0, 0.0, ]))
    
def test_lines_b(three_line_segments):
    _, linesb = three_line_segments
    assert len(linesb) == 3
    assert all(isinstance(line, SegmentLine) for line in linesb)
    
    assert linesb[0].to_array().shape == (17,)  # type + point(3) + direction(3) + num_endpoints + end_point 0 + cos_feature(6)
    assert np.allclose(linesb[0].to_array(), 
                       np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0, 0.0, 0.0,]))
    
    assert linesb[1].to_array().shape == (20,)
    assert np.allclose(linesb[1].to_array(), 
                       np.array([1.0, 0.0, 0.0, 0.0, 1.0/np.sqrt(2), 0.0, 1.0/np.sqrt(2), 2.0, -1.0, 0.0, -1.0, 2.0, 0.0, 2.0, np.sqrt(2)/2, 0.0, np.sqrt(2)/2, 0.0, 0.0, 0.0,]))
    
    assert linesb[2].to_array().shape == (14,)
    assert np.allclose(linesb[2].to_array(), 
                       np.array([1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0, 0.0, ]))
    
# auto-generated... not sure this is right?
def test_lines_a_transformed(three_line_segments):
    linesa, _ = three_line_segments
    for line in linesa:
        line.transform(rdpt.xyz_rpy_to_transform(np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, np.pi/6])))
        
    assert len(linesa) == 3
    assert all(isinstance(line, SegmentLine) for line in linesa)
    
    assert linesa[0].to_array().shape == (14,)  # type + point(3) + direction(3) + num_endpoints + cos_feature(6)
    assert np.allclose(linesa[0].to_array()[-6:], 
                       np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0, 0.0, 0.0,]))
    
    assert linesa[1].to_array().shape == (14,)
    assert np.allclose(linesa[1].to_array()[-6:], 
                       np.array([np.sqrt(2)/2, 0.0, np.sqrt(2)/2, 0.0, 0.0, 0.0,]))
    
    assert linesa[2].to_array().shape == (14,)
    assert np.allclose(linesa[2].to_array()[-6:], 
                       np.array([0.0, np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0, 0.0, ]))