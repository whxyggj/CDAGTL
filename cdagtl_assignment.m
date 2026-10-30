function assignment=cdagtl_assignment(scores)
% Maximum-weight square assignment by primal-dual augmenting paths.
% Pads a rectangular score matrix with zeros. Requires only base MATLAB.
assert(isnumeric(scores)&&ismatrix(scores)&&~isempty(scores)&&all(isfinite(scores(:))),'CDAGTL:assignment','Scores must be a nonempty finite matrix.');
n=max(size(scores));S=zeros(n);S(1:size(scores,1),1:size(scores,2))=scores;
cost=max(S(:))-S;u=zeros(n+1,1);v=u;p=u;way=u;
for row=1:n
    p(1)=row;j0=1;minimum=inf(n+1,1);used=false(n+1,1);
    while true
        used(j0)=true;i0=p(j0);delta=Inf;j1=0;
        for j=2:n+1
            if ~used(j)
                reduced=cost(i0,j-1)-u(i0+1)-v(j);
                if reduced<minimum(j),minimum(j)=reduced;way(j)=j0;end
                if minimum(j)<delta,delta=minimum(j);j1=j;end
            end
        end
        for j=1:n+1
            if used(j),u(p(j)+1)=u(p(j)+1)+delta;v(j)=v(j)-delta;
            else,minimum(j)=minimum(j)-delta;end
        end
        j0=j1;if p(j0)==0,break;end
    end
    while j0~=1
        j1=way(j0);p(j0)=p(j1);j0=j1;
    end
end
assignment=zeros(n,1);for j=2:n+1,assignment(p(j))=j-1;end
end
