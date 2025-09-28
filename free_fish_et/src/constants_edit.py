import torch


"""
Coordinate system conventions:

.    openCV: 
.            
.          .´ [Z camera: positive Z look-at]                                                                       .´ Z
.        .´                                                                                                      .´    
.        0/0---X--->                                                                                             0---X--->                  
.        |                                                                                                       |                                   
.        |    image:                                                                                             |     world:             
.        Y    Point(x,y) -> column-major                                                                         Y     right-handed coordinate system, y pointing down
.        |    Mat(y,x)   -> row-major                                                                            |     axes around positive x are clockwise Y, Z
.        |                                                                                                       |     
.        v                     ^    __                                                                           v        ^ 
.                              |   |`.                                                                                    |
.               (1,  0,  0)    |      `.                                                                                  |    (1,  0,  0)
.               (0, -1,  0)----+        `.                                                                                +----(0,  0, -1)     = 90° ccw about x
.               (0,  0, -1)    |          `.                   K @ Blcam2cvcam @ Rt                                       |    (0,  1,  0)
.               Blcam2cvcam    |            `.                                                                            |  Blworld2cvworld
.                              |              `.___________________________________________________________________       |                                              
.    Blender:                                                                                                      `.                    
.                                                                                                                    `.                   
.        ^                                                                                                       ^     world:
.        |                                    <---------------------------.----------------.------------         |     right-handed coordinate system, z pointing up                     
.        |    image:                                                    .´                  `.                   |     axes around positive x are clockwise Y, Z             
.        Y    Vector(x,y) -> column-major                      (  a_w,  skew,  u_0)   ( Rxx,  Rxy,  Rxz,  tx)    Z
.        |    Matrix(y,x) -> row-major                         (   0 ,   a_u,  v_0) @ ( Ryx,  Ryy,  Ryz,  ty)    |   .´ Y
.        |                                                     (   0 ,    0 ,   1 )   ( Rzx,  Rzy,  Rzz,  tz)    | .´
.        0/0---X--->                                                     K                      Rt               0---X--->
.      .´
.    .´ [Z camera: negative Z look-at]   
.
.
.
.
.
.
.    PyTorch3D:
.
.        ^
.        |
.        |
.        Y
.        |  .´ [Z camera: positive Z look-at]
.        |.´                                 
.<---X---0                                      

"""
BLENDERCAM_2_CV = torch.tensor([    
    [ 1,  0,  0 ],
    [ 0, -1,  0 ],
    [ 0,  0, -1 ]
])

BLENDERWORLD_2_CV = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])

CV_2_BLENDERWORLD = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])

BLENDERCAM_2_PYTORCH3D = torch.tensor([    
    [-1,  0,  0 ],
    [ 0,  1,  0 ],
    [ 0,  0, -1 ]
])

BLENDERWORLD_2_PYTORCH3D = torch.tensor([    
    [-1,  0,  0 ],
    [ 0,  0,  1 ],
    [ 0,  1,  0 ]
])

PYTORCH3D_2_BLENDERWORLD = torch.tensor([    
    [-1,  0,  0 ],
    [ 0,  0,  1 ],
    [ 0,  1,  0 ]
])

CV_2_PYTORCH3D = torch.tensor([
    [-1,  0,  0 ],
    [ 0, -1,  0 ],
    [ 0,  0,  1 ]
])
