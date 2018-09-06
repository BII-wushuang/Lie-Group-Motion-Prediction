% Constructs the SE(3) matrix given lie parameters
function SEmatrix = lietomatrix_euler(euler,trans)
    R = [1 0 0; 0 cos(euler(1)) -sin(euler(1)); 0 sin(euler(1)) cos(euler(1))] * [cos(euler(2)) 0 sin(euler(2)); 0 1 0; -sin(euler(2)) 0 cos(euler(2))] * [cos(euler(3)) -sin(euler(3)) 0; sin(euler(3)) cos(euler(3)) 0; 0 0 1];
    T = trans;
   SEmatrix = [R T; 0 0 0 1];
end