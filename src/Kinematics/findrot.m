function axisangle = findrot(u,v)
    w = cross(u,v);
    if(norm(w)<1e-10)
        axisangle = [0;0;0];
    else
        axisangle = w/norm(w)*acos(dot(u,v));
    end
end
