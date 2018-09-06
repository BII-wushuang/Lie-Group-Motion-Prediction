% Converts the rotation matrix to axis-angle representation
function omega = axis_angle(R)
    theta = acos((trace(R)-1)/2);
    if(theta<1e-10)
	omega = [0;0;0];
    else
        omega = theta/(2*sin(theta)) * [R(3,2) - R(2,3); R(1,3) - R(3,1); R(2,1) - R(1,2)];
    end
end
