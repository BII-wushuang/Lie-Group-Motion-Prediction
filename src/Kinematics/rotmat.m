% Converts the axis-angle representation to rotation matrix
function R = rotmat(A)
    theta = norm(A);
    if (norm(A) == 0)
        R = eye(3);
    else
        A = A / norm(A);
        cross_matrix = [0 -A(3) A(2); A(3) 0 -A(1); -A(2) A(1) 0];
        R = eye(3) + sin(theta) * cross_matrix + (1-cos(theta)) * cross_matrix * cross_matrix;
    end
end