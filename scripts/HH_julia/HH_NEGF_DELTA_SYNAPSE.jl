#
# Hodgkin-Huxley Neuron model Nonequilibrium Green Functions
#
# Gabriel Marghoti / 2024 - UFPR - Curitiba - Brazil
#
using Plots
using DifferentialEquations
using LaTeXStrings
using ProgressBars
using Statistics
using Measures

gr()

function Θ(x)
    if x<0
        return 0.0
    else
        return 1.0
    end
end

function Φ(Veq, β, V_th)
    return 1/(1 + exp(-β*(Veq-V_th)))
end
function dΦ(Veq, β, V_th)
    return ((β*exp(-β*(Veq-V_th)))/(1 + exp(-β*(Veq-V_th)))^2)
end


function conv(X, Y, interval)
    return ((interval[end]-interval[1])/length(interval))*sum(X.*Y)
end
#=
α_m(V) = 0.1 * (V+25.0) / (exp((V+25.0)/10) - 1.0)
β_m(V) = 4.0 * exp((V) / 18.0)

α_h(V) = 0.07 * exp((V) / 20.0)
β_h(V) = 1.0 / (exp((V+30.0)/10) + 1.0)

α_n(V) = 0.01 * (V+10.0) / (exp((V+10.0)/10) - 1.0)
β_n(V) = 0.125 * exp(V / 80.0)
=#

α_m(V) = 0.1 * (V+40.0) / (1.0 - exp(-0.1 * (V+40.0)))
β_m(V) = 4.0 * exp(-(V+65.0) / 18.0)
α_h(V) = 0.07 * exp(-(V+65.0) / 20.0)
β_h(V) = 1.0 / (1.0 + exp(-0.1 * (V+35.0)))
α_n(V) = 0.01 * (V+55.0) / (1.0 - exp(-0.1 * (V+55.0)))
β_n(V) = 0.125 * exp(-(V+65.0) / 80.0)

# Hodgkin-Huxley model equations
function hh_model!(du, u, p, t)
    
    # Load parameters in the integrator step function
    C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, N, A, B, Es, a_r, a_d, β, V_th, I_ext, stim = p

    for i=1:N # Iteration over neurons variables
        V, m, h, n = u[i, 1:4]

        I_ext = 0.0
        if i == 1 && (t>=4.0 && t<=5.0) # || (t>=10.0 && t<=11.0) || (t>=20.0 && t<=21.0) # Adds external stimulation
            I_ext += stim
        end

        for j=1:N # Coupling terms from pre-synaptic neurons
            I_ext += -A[i, j]*(u[i,1] - u[j,1]) - B[i,j]*u[i,4+j]*(u[i, 1] - Es[i,j])
            du[i, 4+j] = (a_r[i,j]*(1/(1 + exp(-β[i,j]*(u[j,1]-V_th[i,j]))))*(1-u[i, 4+j])-a_d[i,j]*u[i, 4+j])
        end
        
        du[i, 1] = (I_ext - g_Na_bar*m^3*h*(V - E_Na) - g_K_bar*n^4*(V - E_K) - g_L_bar*(V - E_L)) / C_m
        du[i, 2] = α_m(V) * (1.0 - m) - β_m(V) * m
        du[i, 3] = α_h(V) * (1.0 - h) - β_h(V) * h
        du[i, 4] = α_n(V) * (1.0 - n) - β_n(V) * n
    end
end


function main()
    
    # Time span
    ti =   0.0
    tf =  50.0
    
    resolution = 1200 

    tspan = (ti, tf)

    tₛ = range(ti, tf, length=resolution)

    # Neuron Parameters
    # Define constants
    C_m      =   1.0         # membrane capacitance, in uF/cm^2
    g_Na_bar = 120.0         # maximum conductances, in mS/cm^2
    g_K_bar  =  36.0
    g_L_bar  =   0.3
    E_Na     =  50.0         # Nernst reversal potentials, in mV
    E_K      = -77.0
    E_L      = -55.0 

    # Synapses parameters
    N    = 4 # Number of neurons
    
    # Coupling matrices, adjacency matrices
    A    = [ 0.0  0.0  0.0  0.0;  # S/F   # Gap junction coupling  
             1.0  0.0  0.0  0.0;          # sets the linear coupling topology        4 ← 3    2 ← 1
             0.0  0.0  0.0  0.0;
             0.0  0.0  1.0  0.0]   

    B    = [ 0.0  0.0  0.0  0.0;  # S/F    # Chemical synapse coupling
             0.0  0.0  0.0  0.0;           # sets the nonlinear coupling topology    4    3 ⇐  2   1  
             0.0  1.0  0.0  0.0;
             0.0  0.0  0.0  0.0] 

    Es   = fill(0.0, (N,N))        # mV  # Synapse Nerst potential, determines excitation or inhibition 
    #Es[4, 1] = -85                 # Change this to control Inhibition(-70 value) depends on the synaptic receptor ion potential
    a_r  = fill(5.0, (N,N))        # rates for synaptic channel conductance activity  
    a_d  = fill(1.0, (N,N))
    #a_r[4, 1] = 1.0
    β    = fill(0.125, (N,N))      # (mV)⁻¹
    V_th = fill(-50.0  , (N,N))    #  mV    # update when having the equilibrium values
    V_th[3, 2] = -10.0             #  mV    # ϕ(V=V_th) = 1/2

    # External current
    I_ext = 0.0  # in μA/cm^2     
    stim  = 10.0  # in μA/cm^2

                
    ts_plot = [10; 20; 40; 80; 120; 200; 280; 400; 500; 600; 1000]

    figures_path = "/home/gabriel/figuras/qualification/NEGF_HH_DELTA_SYNAPSE/$(g_Na_bar)_$(g_K_bar)_$(g_L_bar)_$(E_Na)_$(E_K)_$(E_L)_syn_$(minimum(Es))_stim_$(I_ext)_$(stim)/"
    
    for i=1:N
        mkpath(figures_path*"/neuron$(i)/")
    end

    u0 = rand(N, N+4)

    V0 = zeros(N)
    m0 = zeros(N)
    h0 = zeros(N)
    n0 = zeros(N)

    gS0 = zeros(N, N)

    for itr=1:10 #iteraction to remove transient and set parameter V_th equals to the equilibrium values V0
    # Solve the differential equations TRANSIENT
        prob = ODEProblem(hh_model!, u0, tspan, [C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, N, A, B, Es, a_r, a_d, β, V_th, I_ext, 0.0])
        sol = solve(prob, Tsit5(),  dt=0.001, saveat=tₛ, reltol=1e-9, abstol=1e-9, maxiters = 1e7)

    #    png(plot(sol, layout=(4,1), lc=:black, xlabel=["" "" "" "Time (ms)"], ylabel=["V(t) (mV)" "m(t)" "h(t)" "n(t)"], label="", frame_style=:box, size=(500,500), dpi=200), figures_path*"neuron$(i)/HH_variables_simulation_transient")

        u0 = sol[end]
        
        for i=1:N
            V0[i], m0[i], h0[i], n0[i] = u0[i,1:4]
        end
        
        gS0 = u0[:, 5:end]

        V_th = [fill(V0[1], N) fill(V0[2], N) fill(V0[3], N) fill(V0[4], N)] 
        V_th[3, 2] = -10.0             #  mV    # ϕ(V=V_th) = 1/2
        V_th[4, 1] = -10.0             #  mV    # ϕ(V=V_th) = 1/2
    end

    # Solve the differential equations STIMULATED
    prob = ODEProblem(hh_model!, u0, tspan, [C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, N, A, B, Es, a_r, a_d, β, V_th, I_ext, stim])
    sol = solve(prob, Tsit5(),  dt=0.001, saveat=tₛ, reltol=1e-9, abstol=1e-9, maxiters = 1e7)

    V = zeros(resolution, N)
    m = zeros(resolution, N)
    h = zeros(resolution, N)
    n = zeros(resolution, N)

    gS = zeros(resolution, N, N)

    for t=1:resolution
        V[t, :]     = sol[t][:, 1]
        m[t, :]     = sol[t][:, 2]
        h[t, :]     = sol[t][:, 3]
        n[t, :]     = sol[t][:, 4]
        gS[t, :, :] = sol[t][:, 5:end]
    end
    
    ΔV = zeros(resolution, N)
    ΔgS = zeros((resolution, N, N))

    for t=1:resolution
        ΔV[t, :] = V[t, :] .- V0
        ΔgS[t, :, :] = gS[t, :, :] .- gS0
    end

    n_inf = α_n.(V) ./ (α_n.(V) .+ β_n.(V))
    tau_n = 1. ./ (α_n.(V) .+ β_n.(V))
    n_inf0 = α_n.(V[1, :]) ./ (α_n.(V[1, :]) .+ β_n.(V[1, :]))

    m_inf = α_m.(V) ./ (α_m.(V) .+ β_m.(V))
    tau_m = 1. ./ (α_m.(V) .+ β_m.(V))
    m_inf0 = α_m.(V[1, :]) ./ (α_m.(V[1, :]) .+ β_m.(V[1, :]))

    h_inf = α_h.(V) ./ (α_h.(V) .+ β_h.(V))
    tau_h = 1. ./ (α_h.(V) .+ β_h.(V))
    h_inf0 = α_h.(V[1, :]) ./ (α_h.(V[1, :]) .+ β_h.(V[1, :]))

    for i=1:N
        var_plots = plot(sol.t, [V[:, i] m[:, i] h[:, i] n[:, i]], layout=(4,1), 
        lc=:black, xlabel=["" "" "" "Time (ms)"], ylabel=["V(t) (mV)" "m(t)" "h(t)" "n(t)"], label="", 
        frame_style=:box, size=(900,500), dpi=200, grid=false)
        hline!([V0[i] m0[i] h0[i] n0[i]], label="", lc=:black, ls=:dash)
        
        png(var_plots, figures_path*"neuron$(i)/HH_variables_simulation_neurons")
            
        var_syn_plots = plot(sol.t, [gS[:, i,1] gS[:, i,2] gS[:, i,3] gS[:, i,4]], layout=(4,1), 
            lc=:black, xlabel=["" "" "" "Time (ms)"], ylabel=["gS$(i)1" "gS$(i)2" "gS$(i)3" "gS$(i)4"], label="", 
            frame_style=:box, size=(900,500), dpi=200, grid=false)
            hline!([gS0[i,1] gS0[i,2] gS0[i,3] gS0[i,4]], label="", lc=:black, ls=:dash)
        
        png(var_syn_plots,figures_path*"neuron$(i)/Synaptic_var_gS_post_syn")


        cond_plote = plot(tₛ, [V[:, i] (g_Na_bar*m[:, i].^(3).*h[:, i]) (g_K_bar*n[:, i].^4)], layout=(3,1), 
        lc=:black, xlabel=["" "" L"Time\ (ms)"], ylabel=[L"V(t)\ (mV)" L"g_{Na} (t)" L"g_{K} (t)"], label="", 
        frame_style=:box, size=(900,500), dpi=200, left_margin=5mm, grid=false)
        hline!([V0[i] (g_Na_bar*m0[i].^(3).*h0[i]) (g_K_bar*n0[i].^4)], label=["Equilibrium" "" "" ""], lc=:red, ls=:dash)
        
        png(cond_plote, figures_path*"neuron$(i)/HH_conductances_simulation")

        png(plot(tₛ, [(tau_n[:, i]) (tau_m[:, i]) (tau_h[:, i])], layout=(3,1), 
        lc=:black, xlabel=["" "" L"Time\ (ms)"], ylabel=[L"n" L"m" L"h"], label="", 
        frame_style=:box, size=(500,500), dpi=200, grid=false),
        figures_path*"neuron$(i)/HH_taus")

        png(plot(tₛ, [(n_inf[:, i]) (m_inf[:, i]) (h_inf[:, i])], layout=(3,1), 
        lc=:black, xlabel=["" "" L"Time\ (ms)"], ylabel=[L"n" L"m" L"h"], label="", 
        frame_style=:box, size=(500,500), dpi=200, grid=false),
        figures_path*"neuron$(i)/HH_saturation variables")
    end

    #### Equilibrium direct green functions
    σ₀  = zeros(resolution, resolution, N, N)
    gg₀ = zeros(resolution, resolution, N, N)
    gs₀ = zeros(resolution, resolution, N, N)
    g₀  = zeros(resolution, resolution, N, N)

    for j=1:N
        for i=j:N
            for t′=1:resolution
                σ₀[:, t′, i, j]  = Θ.(tₛ.-tₛ[t′]) .*a_r[i,j]*(1-gS0[i, j]).*dΦ(V0[j], β[i,j], V_th[i,j]).*exp.(-(tₛ.-tₛ[t′]).*(a_d[i,j]-a_r[i,j]/(1 + exp(-β[i,j]*(V0[j]-V_th[i,j])))))
                gg₀[:, t′, i, j] = Θ.(tₛ.-tₛ[t′]) .*A[i, j].*exp.(-(tₛ.-tₛ[t′]).*(g_L_bar.+sum(A[i, :]).+sum(B[i, :].*gS0[i, :])))
                gs₀[:, t′, i, j] = Θ.(tₛ.-tₛ[t′]) .*B[i, j].*(Es[i,j]-V0[i]).*exp.(-(tₛ.-tₛ[t′]).*(g_L_bar.+sum(A[i, :]).+sum(B[i, :].*gS0[i, :])))
            end
            png(
                heatmap(tₛ, tₛ, σ₀[:, :, i, j],
                    xflip=false,
                    ylabel="t (current time)",
                    xlabel = "t′ (past time)",
                    size=(500,400),
                    fillcolor=:jet1,
                    dpi=200, 
                    frame_style=:box,
                    grid=false),
                figures_path*"neuron$(i)/sigmaZERO_ij$(i)_$(j)"
            )

            png(heatmap(tₛ, tₛ, gs₀[:, :, i, j],
            xflip=false,
            ylabel="t (current time)",
            xlabel = "t′ (past time)",
            size=(500,400),
            fillcolor=:jet1,
            dpi=200, 
            frame_style=:box,
            grid=false),
                figures_path*"neuron$(i)/gsZERO_ij$(i)_$(j)"
            )

            png(heatmap(tₛ, tₛ, gg₀[:, :, i, j],
            xflip=false,
            ylabel="t (current time)",
            xlabel = "t′ (past time)",
            size=(500,400),
            fillcolor=:jet1,
            dpi=200, 
            frame_style=:box,
            grid=false),
                figures_path*"neuron$(i)/ggZERO_ij$(i)_$(j)"
            )
        end
    end

    for t=1:resolution
        for t′=1:(t-1)
            for i=1:N
                for j=1:N
                    g₀[t, t′, i, j] = gg₀[t, t′, i, j] + conv(gs₀[t, t′:t, i, j], σ₀[t′:t, t′, i, j], tₛ[t′:t])
                end
            end
        end
    end


#   NONequilibrium green functions
    σ_n_V  = zeros(resolution, resolution, N)
    κ_gK_V = zeros(resolution, resolution, N)
    
    σ_m_V  = zeros(resolution, resolution, N)
    σ_h_V  = zeros(resolution, resolution, N)

    κ_gNa_V = zeros(resolution, resolution, N)
    
    conv_sigma_n_V = zeros(resolution, N) # n approx
    conv_sigma_m_V = zeros(resolution, N) # m approx
    conv_sigma_h_V = zeros(resolution, N) # h approx

    tau_i = (C_m)./(g_L_bar.+(g_Na_bar*m0.^(3).*h0).+(g_K_bar*n0.^4))

    
    Χ_K  = zeros(resolution, resolution, N)
    Χ_Na = zeros(resolution, resolution, N)

    Χ_i  = zeros(resolution, resolution, N)

    for itr=ProgressBar(1:10)  #### iterative method to approximate σ
        for i=1:N
            for t=1:resolution
                for t′=1:(t) 
                    nume_n = n_inf[t′, i].- (n_inf0[i] .+ (conv_sigma_n_V[t′, i]))
                    nume_m = m_inf[t′, i].- (m_inf0[i] .+ (conv_sigma_m_V[t′, i]))
                    nume_h = h_inf[t′, i].- (h_inf0[i] .+ (conv_sigma_h_V[t′, i]))
                    if nume_n == 0.0
                        σ_n_V[t, t′, i]  = 0.0
                    elseif abs(tau_n[t′])>=1.0e-1
                        σ_n_V[t, t′, i]  = nume_n / (tau_n[t′, i]*(V[t′, i].-V0[i]))
                    end
                    if nume_m == 0.0
                        σ_m_V[t, t′, i]  = 0.0
                    else
                        σ_m_V[t, t′, i]  = nume_m / (tau_m[t′, i]*(V[t′, i].-V0[i]))
                    end
                    if nume_h == 0.0
                        σ_h_V[t, t′, i]  = 0.0
                    else
                        σ_h_V[t, t′, i]  = nume_h / (tau_h[t′, i]*(V[t′, i].-V0[i]))
                    end

                    κ_gK_V[t, t′, i] = g_K_bar*(conv_sigma_n_V[t′, i]^3)*((n_inf[t′, i]- n_inf0[i]) - (conv_sigma_n_V[t′, i]))/(tau_m[t′, i]*(V[t′, i].-V0[i]))

                    κ_gNa_V[t, t′, i] = (g_K_bar*conv_sigma_m_V[t′, i]^2)*(conv_sigma_h_V[t′, i]*(((m_inf[t′, i]- m_inf0[i]) - conv_sigma_m_V[t′, i])/(tau_m[t′, i]*(V[t′, i].-V0[i]))) + conv_sigma_m_V[t′, i]*(((h_inf[t′, i]- h_inf0[i]) - conv_sigma_h_V[t′, i])/(tau_h[t′, i]*(V[t′, i].-V0[i]))))

                    Χ_K[t, t′, i]  = -exp(-(t-t′)/tau_i[i])*(V[t′, i] - E_K )/C_m
                    Χ_Na[t, t′, i] = -exp(-(t-t′)/tau_i[i])*(V[t′, i] - E_Na)/C_m
                end
                conv_sigma_n_V[t, i] = conv(σ_n_V[t, 1:t, i],(V[1:t, i].-V0[i]), tₛ[1:t])
                conv_sigma_m_V[t, i] = conv(σ_m_V[t, 1:t, i],(V[1:t, i].-V0[i]), tₛ[1:t])
                if  conv_sigma_m_V[t, i]>=1.0 && t>1
                    conv_sigma_m_V[t, i] = conv_sigma_m_V[t-rand(1:(t-1)), i]
                end
                conv_sigma_h_V[t, i] = conv(σ_h_V[t, 1:t, i], (V[1:t, i].-V0[i]), tₛ[1:t])
                for t′=1:t
                    Χ_i[t, t′, i] = conv(Χ_K[t, t′:t, i], κ_gK_V[t′:t, t′, i], tₛ[t′:t]) + conv(Χ_Na[t, t′:t, i], κ_gNa_V[t′:t, t′, i], tₛ[t′:t])
                end
            end

            if itr%10==0 || itr==1
                png(
                    heatmap(tₛ, tₛ, Χ_i[:, :, i],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title= "χᵢ(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            clims = (-1.0,1.0),
                            right_margin=2mm,
                            grid=false),
                            figures_path*"neuron$(i)/Χ_i"
                    )

                    plote = plot()
                    
                    vline!([4; 5],
                    lw=1,
                    la=1.0,
                    label="",
                    lc=[:gray],
                    ls=[:solid],
                    annotation=(2.5, 1.8, "Stimulus")
                    )
                    for t_plot in ts_plot
                        plot!(tₛ[1:(t_plot)] , Χ_i[t_plot, 1:(t_plot), i],
                        ylabel= "χᵢ(t,t′)",
                        label="t=$(round(tₛ[t_plot],digits=1))",
                        xlabel = "t′ (ms)",
                        size=(900,600),
                        lc=RGBA(1-t_plot/resolution, 0, t_plot/resolution, 1),
                        ls=[:solid],
                        lw=2.5,
                        dpi=200, 
                        frame_style=:box,
                        ylims = (-1.0,1.0),
                        xlims=(0,tₛ[end]),
                        left_margin=2mm,
                        grid=false)
                    end
                
                    plot!(tₛ[1:(end)] , Χ_i[end, 1:(end), i],
                    ylabel= "χᵢ(t,t′)",
                    label="t=$(round(tₛ[end],digits=1))",
                    xlabel = "t′ (ms)",
                    size=(1000,800),
                    lw=1,
                    la=1.0,
                    lc=[:black],
                    ls=[:solid],
                    dpi=200, 
                    frame_style=:box,
                    #ylims = (-0.1,0.1),
                    grid=false)
                        png(plote,figures_path*"neuron$(i)/level_CHI_i")



                png(
                    heatmap(tₛ, tₛ, σ_n_V[:, :, i],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title=L"σ_{n,V}(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            clims = (-0.1,0.1),
                            grid=false),
                            figures_path*"neuron$(i)/sigma_n_V"
                    )
                    png(
                        heatmap(tₛ, tₛ, κ_gK_V[:, :, i],
                                xflip=false,
                                ylabel="t (current time)",
                                xlabel = "t′ (past time)",
                                size=(500,400),
                                title=L"κ_{K,V}(t,t′)",
                                fillcolor=:jet1,
                                dpi=200, 
                                frame_style=:box,
                                clims = (-0.1,0.1),
                                grid=false),
                                figures_path*"neuron$(i)/Gamma_n_V"
                        )

                png(
                    heatmap(tₛ, tₛ, σ_m_V[:, :, i],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title=L"σ_{m,V}(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            clims = (-0.1,0.1),
                            grid=false),
                            figures_path*"neuron$(i)/sigma_m_V"
                    )
        
                png(
                    heatmap(tₛ, tₛ, σ_h_V[:, :, i],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title=L"σ_{h,V}(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            clims = (-0.1,0.1),
                            grid=false),
                            figures_path*"neuron$(i)/sigma_h_V"
                    )

                png(
                    plot(tₛ[end] .- tₛ, σ_n_V[end, 1:end, i],
                            xflip=false,
                            ylabel= [L"σ_{n,V}(t,t′)"],
                            label="",
                            xlabel = "t - t′",
                            title="t=$(tₛ[end])",
                            size=(1000,800),
                            lc=[:black :red],
                            ls=[:solid :dash],
                            dpi=200, 
                            frame_style=:box,
                            #ylims = (-0.1,0.1),
                            grid=false),
                            figures_path*"neuron$(i)/level_curve_sigma_n_V"
                    )
                    png(
                        plot(tₛ[end] .- tₛ, [κ_gK_V[end, 1:end, i] ],
                                xflip=false,
                                ylabel= [L"κ_{K,V}(t,t′)"],
                                label="",
                                xlabel = "t - t′",
                                title="t=$(tₛ[end])",
                                size=(1000,800),
                                lc=[:black :red],
                                ls=[:solid :dash],
                                dpi=200, 
                                frame_style=:box,
                                #ylims = (-0.1,0.1),
                                grid=false),
                                figures_path*"neuron$(i)/level_curve_Gamma_K_V"
                        )
                        png(
                            plot(tₛ[end] .- tₛ, [κ_gNa_V[end, 1:end, i] ],
                                    xflip=false,
                                    ylabel= [L"κ_{Na,V}(t,t′)"],
                                    label="",
                                    xlabel = "t - t′",
                                    title="t=$(tₛ[end])",
                                    size=(1000,800),
                                    lc=[:black :red],
                                    ls=[:solid :dash],
                                    dpi=200, 
                                    frame_style=:box,
                                    #ylims = (-0.1,0.1),
                                    grid=false),
                                    figures_path*"neuron$(i)/level_curve_Gamma_Na_V"
                            )
                    png(
                        plot(tₛ, [σ_m_V[end, 1:end, i] ],
                                xflip=false,
                                ylabel= [L"σ_{m,V}(t,t′)"],
                                label="",
                                xlabel = "t - t′",
                                title="t=$(tₛ[end])",
                                size=(1000,800),
                                lc=[:black :red],
                                ls=[:solid :dash],
                                dpi=200, 
                                frame_style=:box,
                                ylims = (-0.1,0.1),
                                grid=false),
                                figures_path*"neuron$(i)/level_curve_sigma_m_V"
                        )
                        png(
                            plot(tₛ[end] .- tₛ, [σ_m_V[end, 1:end, i] ],
                                    xflip=false,
                                    ylabel= [L"σ_{h,V}(t,t′)"],
                                    label="",
                                    xlabel = "t - t′",
                                    title="t=$(tₛ[end])",
                                    size=(1000,800),
                                    lc=[:black :red],
                                    ls=[:solid :dash],
                                    dpi=200, 
                                    frame_style=:box,
                                    ylims = (-0.1,0.1),
                                    grid=false),
                                    figures_path*"neuron$(i)/level_curve_sigma_h_V"
                            )
                plot_conv_n_V = plot(tₛ, [n[:, i].-n0[i] conv_sigma_n_V[:, i]],     
                label=["Original" "Estimated"],
                size=(500,400),
                lc = [:black :red],
                ls = [:solid :dash],
                xlabel="Time (ms)",
                ylabel = "n",
                dpi=200, 
                frame_style=:box,
                ylims=(-0.01, 0.51),
                grid=false)
                png(plot_conv_n_V,       
                    figures_path*"neuron$(i)/plot_n_simulated_estimated_itr$(itr)"
                )
                plot_conv_n_V = plot(tₛ, [m[:, i].-m0[i] conv_sigma_m_V[:, i]],     
                label=["Original" "Estimated"],
                size=(500,400),
                lc = [:black :red],
                ls = [:solid :dash],
                xlabel="Time (ms)",
                ylabel = "m",
                dpi=200, 
                frame_style=:box,
                ylims=(-0.11, 1.01),
                grid=false)
                png(plot_conv_n_V,       
                    figures_path*"neuron$(i)/plot_m_simulated_estimated_itr$(itr)"
                )
                plot_conv_n_V = plot(tₛ, [h[:, i].-h0[i] conv_sigma_h_V[:, i]],     
                label=["Original" "Estimated"],
                size=(500,400),
                lc = [:black :red],
                ls = [:solid :dash],
                xlabel="Time (ms)",
                ylabel = "h",
                dpi=200, 
                frame_style=:box,
                ylims=(-1.01, 0.11),
                grid=false)
                png(plot_conv_n_V,       
                    figures_path*"neuron$(i)/plot_h_simulated_estimated_itr$(itr)"
                )
            end
        end
    end


    ###############################################
    #### NONequilibrium interaction Green Functions

    σ = zeros(resolution, resolution, N, N)
    𝜋 = zeros(resolution, resolution, N, N)
    g = zeros(resolution, resolution, N, N)


    conv_sigma_V = zeros(resolution, N, N) # ΔS
    for itr=1:10  #### iterative method to approximate σ
        for j=1:N
            for i=j:N
                for t=1:resolution
                    for t′=1:(t-1)   
                        if V[t′, j] == V0[j]
                            σ[t, t′, i, j]  = 0.0 #(σ₀[t, t′, i, j]/dΦ(V0[j], β[i, j], V_th[i, j]))*0*(1 - (Ss[t′, i, j]-gS0[i, j])/(1-gS0[i, j]))
                        else
                            σ[t, t′, i, j]  = (σ₀[t, t′, i, j]/dΦ(V0[j], β[i, j], V_th[i, j]))*((Φ(V[t′, j], β[i, j], V_th[i,j])-Φ(V0[j], β[i, j], V_th[i,j]))/ΔV[t′, j])*(1 - (conv_sigma_V[t′, i, j])/(1-gS0[i, j]))
                        end
                    end
                end
                for t=1:resolution
                    conv_sigma_V[t, i, j] = conv(σ[t, 1:t, i, j], ΔV[1:t, j], tₛ[1:t])
                end
                        
            for t=1:resolution
                for t′=1:(t-1) 
                    𝜋[t, t′, i, j] = conv(gs₀[t, t′:t, i, j], (1 .-(ΔV[t′:t, i]/(Es[i, j]-V0[i]))).*σ[t′:t, t′, i, j], tₛ[t′:t])
                    g[t, t′, i, j] = gg₀[t, t′, i, j] + 𝜋[t, t′, i, j]
                end
            end
                    
                png(
                    heatmap(tₛ, tₛ, σ[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title="σ_$(i),$(j)(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"neuron$(i)/sigma_ij$(i)_$(j)"
                    )

                png(
                    heatmap(tₛ, tₛ, g[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            title="g_$(i),$(j)(t,t′)",
                            size=(500,400),
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"neuron$(i)/g_ij$(i)_$(j)"
                    )
                png(
                    heatmap(tₛ, tₛ, 𝜋[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            title="𝜋_$(i),$(j)(t,t′)",
                            size=(500,400),
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"neuron$(i)/𝜋_ij$(i)_$(j)"
                    )
            end
        end
        plot_Delta_S_sigma_V = plot(tₛ, [ΔgS[:, 2, 1] ΔgS[:, 3, 2] ΔgS[:, 4, 1]  ΔgS[:, 4, 3]],     # plot only the synapses that matter for the networks (4←3←2←1)
        label=["S [2←1]" "S [3⇐2]" "S [4⥒1]" "S [4←3]"],
        size=(500,400),
        lc = [:gray :black :red :blue],
        xlabel="time (ms)",
        ylabel = "ΔS",
        dpi=200, 
        frame_style=:box,
        grid=false)
        plot!(tₛ, [conv_sigma_V[:, 2, 1] conv_sigma_V[:, 3, 2] conv_sigma_V[:, 4, 1]  conv_sigma_V[:, 4, 3]],     # plot only the synapses that matter for the networks (4←3←2←1)
        label="",
        lc = [:gray :black :red :blue],
        ls=:dot,
        lw=2,
        size=(500,400),
        xlabel="time (ms)",
        ylabel = "ΔS",
        dpi=200, 
        frame_style=:box,
        grid=false),
        png(plot_Delta_S_sigma_V,       
            figures_path*"/synapses_duration_stim_itr$(itr)"
        )
    end


        plot_Delta_S_sigma_V = plot(tₛ, [ΔgS[:, 2, 1] ΔgS[:, 3, 2] ΔgS[:, 4, 1]  ΔgS[:, 4, 1]],     # plot only the synapses that matter for the networks (4←3←2←1)
        label=["S [2←1]" "S [3⇐2]" "S [4⥒1]" "S [4←3]"],
        lc = [:gray :black :red :blue],
        size=(500,400),
        xlabel="time (ms)",
        ylabel = "ΔS",
        dpi=200, 
        frame_style=:box,
        grid=false)
        plot!(tₛ, [conv_sigma_V[:, 2, 1] conv_sigma_V[:, 3, 2] conv_sigma_V[:, 4, 1]  conv_sigma_V[:, 4, 3]],     # plot only the synapses that matter for the networks (4←3←2←1)
        label="",
        lc = [:gray :black :red :blue],
        ls=:dot,
        lw=2,
        size=(500,400),
        xlabel="time (ms)",
        ylabel = "ΔS",
        dpi=200, 
        frame_style=:box,
        grid=false),
    png(plot_Delta_S_sigma_V,       
        figures_path*"/synapses_duration_stim"
    )

    #######################################
    ##### Interaction Paths Green functions
    #####    
    # The connected Green function is the directed plus the convolution with the paths from node j to node i
    # First compute the trivial ones, the first neighbors of the boundary condition ΔVⱼ, then propagate to higher neighbors 
    G₀ = g₀                # First neighbors (direct)       

    conv_G_V = zeros(resolution, N) # test G function
    for t=1:resolution
        for t′=1:(t-1)         # Second neighbors
            G₀[t, t′, 3, 1] += conv(g₀[t, t′:t, 3, 2], G₀[t′:t, t′, 2, 1], tₛ[t′:t])
            G₀[t, t′, 4, 2] += conv(g₀[t, t′:t, 4, 3], G₀[t′:t, t′, 3, 2], tₛ[t′:t])
        end
    end
    for t=1:resolution    # Third neighbors
        for t′=1:(t-1)
            G₀[t, t′, 4, 1] += conv(g₀[t, t′:t, 4, 3], G₀[t′:t, t′, 3, 1], tₛ[t′:t])
        end
    end

    G = g                # First neighbors (direct)                        
    for t=1:resolution
        for t′=1:(t-1)         # Second neighbors
            G[t, t′, 3, 1] += conv(g[t, t′:t, 3, 2], G[t′:t, t′, 2, 1], tₛ[t′:t])
            G[t, t′, 4, 2] += conv(g[t, t′:t, 4, 3], G[t′:t, t′, 3, 2], tₛ[t′:t])
        end
    end
    for t=1:resolution    # Third neighbors
        for t′=1:(t-1)
            G[t, t′, 4, 1] += conv(g[t, t′:t, 4, 3], G[t′:t, t′, 3, 1], tₛ[t′:t])
        end
    end
    for i=1:N
        for t=1:resolution
            for j=1:N
                conv_G_V[t, i] += conv(g[t, 1:t, i, j], ΔV[1:t, j], tₛ[1:t])
            end
        end
    end


    for j=1:N-1
        for i=j:N
            png(
                heatmap(tₛ, tₛ, G₀[:, :, i, j],
                    xflip=false,
                    ylabel="t (current time)",
                    xlabel = "t′ (past time)",
                    size=(500,400),
                    title="G₀$(i),$(j)(t,t′)",
                    fillcolor=:jet1,
                    dpi=200, 
                    frame_style=:box,
                    grid=false),
                    figures_path*"neuron$(i)/G0_ij$(i)_$(j)"
                )
            png(
                heatmap(tₛ, tₛ, G[:, :, i, j],
                    xflip=false,
                    ylabel="t (current time)",
                    xlabel = "t′ (past time)",
                    title="G$(i),$(j)(t,t′)",
                    size=(500,400),
                    fillcolor=:jet1,
                    dpi=200, 
                    frame_style=:box,
                    grid=false),
                    figures_path*"neuron$(i)/G_ij$(i)_$(j)"
                )

            plote_level_curves_G = plot(tₛ[end].-tₛ[1:end], G₀[end, 1:end, i, j],
                            xlabel="t-t′",
                            ylabel = "G",
                            label= "G₀",
                            lc=:black,
                            lw=2.4,
                            size=(500,400),
                            dpi=200, 
                            frame_style=:box,
                            grid=false)
            for t_idx in ts_plot
                plot!(tₛ[t_idx].-tₛ[2:(t_idx)], G[t_idx, 2:(t_idx), i, j],
                            xlabel="t-t′",
                            ylabel = "G",
                            lc = RGBA(t_idx/ts_plot[end], 0, 1 - (2*t_idx/ts_plot[end] -1)^2, 1),
                            la=0.9,
                            lw=1.4,
                            ls= :solid, #rand([:dash; :dot]),
                            label= "t = "*string(round(tₛ[t_idx], digits=2)),
                            size=(500,400),
                            dpi=200, 
                            frame_style=:box,
                            grid=false)
                    png(plote_level_curves_G,
                            figures_path*"neuron$(i)/G_level_curves_ij$(i)_$(j)"
                        )
            end
        end
    end

#
end

main()
