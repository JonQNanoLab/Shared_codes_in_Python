Xaxis = thetalist
Yaxis = Dplist/1208e-9
Zaxis = log10(fCDSweepN50)


surf(Xaxis, Yaxis, Zaxis, 'EdgeColor', 'None')
colormap jet
colorbar
view(2)

xlabel('\theta (rad)')
ylabel('D´/\lambda')
title('Averaged f_{CD} , N = 50')

xlim([Xaxis(1) Xaxis(length(Xaxis))])
ylim([Yaxis(1) Yaxis(length(Yaxis))])

xticks([0.25 0.5 0.75 1 1.25 1.5])
yticks([0.5 1 1.5 2])

set(gca, 'Layer', 'top')
