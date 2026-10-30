function [C,info]=cdagtl_consensus(C,F,M,alpha,curvature,maxIter,tol)
% Inexact minimization of ORIGINAL C block with projected-gradient/Armijo steps.
% curvature=mu*number_of_views. Reject zero-degree candidates.
fun=@(P) alpha*cdagtl_graph(P,F)+curvature/2*norm(P-M,'fro')^2;
initial=fun(C); info.accepted=false; info.innerIterations=0; step=1/curvature;
for k=1:maxIter
    [~,G]=cdagtl_graph(C,F); G=alpha*G+curvature*(C-M);
    mapping=C-cdagtl_simplex(C-G);
    if norm(mapping,'fro')/max(1,norm(C,'fro'))<=tol,break;end
    current=fun(C); accepted=false;
    for bt=1:60
        trial=cdagtl_simplex(C-step*G); direction=trial-C;
        if all(sum(trial,2)>0)
            candidate=fun(trial);
            if isfinite(candidate)&&candidate<=current+1e-4*sum(G(:).*direction(:))
                C=trial; accepted=true;info.accepted=true;break;
            end
        end
        step=step/2;
    end
    info.innerIterations=k;
    if ~accepted,break;end
    step=min(step*1.5,1/curvature);
end
[~,G]=cdagtl_graph(C,F);G=alpha*G+curvature*(C-M);
info.projectedGradient=norm(C-cdagtl_simplex(C-G),'fro')/max(1,norm(C,'fro'));
info.objectiveChange=fun(C)-initial;
end
