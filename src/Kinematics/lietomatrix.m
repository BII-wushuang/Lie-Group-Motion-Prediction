% Constructs the SE(3) matrix given lie parameters
function SEmatrix = lietomatrix(angles,trans)
    R = rotmat(angles);
    T = trans;
   SEmatrix = [R T; 0 0 0 1];
end