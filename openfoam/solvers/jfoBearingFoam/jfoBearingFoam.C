/*---------------------------------------------------------------------------*\
  Native OpenFOAM finite-volume Reynolds/JFO solver for a conical bearing.
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "calculatedFvPatchFields.H"
#include "fixedValueFvPatchFields.H"
#include "fvcDiv.H"
#include "fvcGrad.H"
#include "fvmLaplacian.H"
#include "solverPerformance.H"
#include "surfaceInterpolate.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Solve the mass-conserving Reynolds/JFO bearing equation"
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    solverPerformance::debug = 0;

    IOdictionary properties
    (
        IOobject
        (
            "jfoProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    const dimensionedScalar length("length", dimLength, properties);
    const dimensionedScalar meanRadius
    (
        "meanRadius",
        dimLength,
        properties
    );
    const dimensionedScalar clearance
    (
        "radialClearance",
        dimLength,
        properties
    );
    const dimensionedScalar eccentricity
    (
        "eccentricity",
        dimLength,
        properties
    );
    const dimensionedScalar feedDiameter
    (
        "feedDiameter",
        dimLength,
        properties
    );
    const dimensionedScalar feedAxialPosition
    (
        "feedAxialPosition",
        dimLength,
        properties
    );
    const dimensionedScalar viscosity
    (
        "dynamicViscosity",
        dimDynamicViscosity,
        properties
    );
    const dimensionedScalar ambientPressure
    (
        "ambientPressure",
        dimPressure,
        properties
    );
    const dimensionedScalar cavitationPressure
    (
        "cavitationPressure",
        dimPressure,
        properties
    );
    const dimensionedScalar feedGaugePressure
    (
        "feedGaugePressure",
        dimPressure,
        properties
    );

    const scalar gamma =
        degToRad(properties.lookup<scalar>("semiconeAngleDegrees"));
    const scalar eccentricityAngle =
        degToRad(properties.lookup<scalar>("eccentricityAngleDegrees"));
    const scalar rpm = properties.lookup<scalar>("rpm");
    const label nTheta = properties.lookup<label>("nTheta");
    const word feedCellZoneName =
        properties.lookupOrDefault<word>("feedCellZone", word::null);
    const label maxRevolutions =
        properties.lookupOrDefault<label>("maxRevolutions", 8);
    const label maxZeroSpeedSteps =
        properties.lookupOrDefault<label>("maxZeroSpeedSteps", 50);
    const label maxActiveIterations =
        properties.lookupOrDefault<label>("maxActiveIterations", 200);
    const scalar convergenceTolerance =
        properties.lookupOrDefault<scalar>("convergenceTolerance", 1e-8);
    const scalar policyTolerance =
        properties.lookupOrDefault<scalar>("policyTolerance", 1e-7);
    const label logInterval =
        properties.lookupOrDefault<label>("logInterval", 32);

    volScalarField p
    (
        IOobject
        (
            "p",
            runTime.name(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );
    volScalarField thetaFill
    (
        IOobject
        (
            "thetaFill",
            runTime.name(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );
    volScalarField filmThickness
    (
        IOobject
        (
            "filmThickness",
            runTime.name(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar(dimLength, 0)
    );
    volScalarField surfaceRadius
    (
        IOobject
        (
            "surfaceRadius",
            runTime.name(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar(dimLength, 0)
    );
    volScalarField surfaceMetric
    (
        IOobject
        (
            "surfaceMetric",
            runTime.name(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar(dimless, 0)
    );

    const scalar cosGamma = Foam::cos(gamma);
    const scalar ex =
        eccentricity.value()*Foam::cos(eccentricityAngle);
    const scalar ey =
        eccentricity.value()*Foam::sin(eccentricityAngle);
    const vectorField& centres = mesh.C();
    boolList feedMask(mesh.nCells(), false);
    DynamicList<label> feedCells;

    forAll(centres, celli)
    {
        const scalar theta = centres[celli].x()/meanRadius.value();
        const scalar z = centres[celli].z();
        const scalar journalRadius =
            meanRadius.value()
          + (0.5*length.value() - z)*Foam::tan(gamma);
        const scalar q =
            ex*Foam::sin(theta) - ey*Foam::cos(theta);
        const scalar journalRayRadius =
            q
          + Foam::sqrt
            (
                sqr(journalRadius)
              - sqr(eccentricity.value())
              + sqr(q)
            );
        const scalar boreRadius = journalRadius + clearance.value();

        filmThickness[celli] =
            (boreRadius - journalRayRadius)*cosGamma;
        surfaceRadius[celli] =
            0.5*(boreRadius + journalRayRadius);
        surfaceMetric[celli] =
            surfaceRadius[celli]/(meanRadius.value()*cosGamma);
    }

    if (feedCellZoneName.empty())
    {
        Info<< "Using legacy geometric cell-centre feed selection" << nl;
        forAll(centres, celli)
        {
            const scalar theta =
                centres[celli].x()/meanRadius.value();
            const scalar z = centres[celli].z();
            scalar wrapped = theta - constant::mathematical::pi;
            while (wrapped > constant::mathematical::pi)
            {
                wrapped -= constant::mathematical::twoPi;
            }
            while (wrapped < -constant::mathematical::pi)
            {
                wrapped += constant::mathematical::twoPi;
            }
            const scalar feedDistance =
                Foam::hypot
                (
                    surfaceRadius[celli]*wrapped,
                    (z - feedAxialPosition.value())/cosGamma
                );
            if (feedDistance <= 0.5*feedDiameter.value())
            {
                feedMask[celli] = true;
                feedCells.append(celli);
            }
        }
    }
    else
    {
        if (!mesh.cellZones().found(feedCellZoneName))
        {
            FatalErrorInFunction
                << "Cannot find required feed cellZone "
                << feedCellZoneName << nl
                << "Available cellZones are " << mesh.cellZones().toc()
                << exit(FatalError);
        }

        const cellZone& feedCellZone =
            mesh.cellZones()[feedCellZoneName];
        forAll(feedCellZone, i)
        {
            const label celli = feedCellZone[i];
            feedMask[celli] = true;
            feedCells.append(celli);
        }
        Info<< "Using topological feed cellZone "
            << feedCellZoneName << nl;
    }

    if (feedCells.empty())
    {
        FatalErrorInFunction
            << "The mesh is too coarse to resolve the feed patch"
            << exit(FatalError);
    }

    forAll(filmThickness.boundaryField(), patchi)
    {
        if
        (
            filmThickness.boundaryField()[patchi].type()
         == calculatedFvPatchScalarField::typeName
        )
        {
            filmThickness.boundaryFieldRef()[patchi] ==
                filmThickness.boundaryField()[patchi].patchInternalField();
            surfaceRadius.boundaryFieldRef()[patchi] ==
                surfaceRadius.boundaryField()[patchi].patchInternalField();
            surfaceMetric.boundaryFieldRef()[patchi] ==
                surfaceMetric.boundaryField()[patchi].patchInternalField();
        }
    }
    filmThickness.correctBoundaryConditions();
    surfaceRadius.correctBoundaryConditions();
    surfaceMetric.correctBoundaryConditions();

    const dimensionSet permeabilityDimensions
    (
        pow3(dimLength)/dimDynamicViscosity
    );
    volSymmTensorField diffusion
    (
        IOobject
        (
            "reynoldsDiffusion",
            runTime.name(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedSymmTensor
        (
            permeabilityDimensions,
            symmTensor::zero
        )
    );

    forAll(diffusion, celli)
    {
        const scalar permeability =
            pow3(filmThickness[celli])/(12*viscosity.value());
        const scalar radialRatio =
            surfaceRadius[celli]/meanRadius.value();
        diffusion[celli] = symmTensor
        (
            permeability/(radialRatio*cosGamma),
            0,
            0,
            0,
            0,
            permeability*radialRatio*cosGamma
        );
    }
    forAll(diffusion.boundaryField(), patchi)
    {
        if
        (
            diffusion.boundaryField()[patchi].type()
         == calculatedFvPatchSymmTensorField::typeName
        )
        {
            diffusion.boundaryFieldRef()[patchi] ==
                diffusion.boundaryField()[patchi].patchInternalField();
        }
    }
    diffusion.correctBoundaryConditions();

    const scalar omega = rpm*constant::mathematical::twoPi/60;
    const dimensionedScalar omegaDimensioned
    (
        "omega",
        dimless/dimTime,
        omega
    );
    const scalar deltaTheta = constant::mathematical::twoPi/nTheta;
    const scalar deltaT =
        mag(omega) < small ? 1 : 2*deltaTheta/mag(omega);
    runTime.setDeltaT(deltaT);

    const scalar feedPressure =
        ambientPressure.value()
      + feedGaugePressure.value()
      - cavitationPressure.value();
    const scalar endPressure =
        ambientPressure.value() - cavitationPressure.value();

    forAll(p.boundaryField(), patchi)
    {
        if
        (
            p.boundaryField()[patchi].type()
         == fixedValueFvPatchScalarField::typeName
        )
        {
            p.boundaryFieldRef()[patchi] == endPressure;
        }
    }
    p.correctBoundaryConditions();

    const surfaceScalarField phiCouette
    (
        IOobject
        (
            "phiCouette",
            runTime.name(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        (omegaDimensioned/(2*cosGamma))
       *fvc::interpolate(surfaceRadius)
       *(mesh.Sf() & vector(1, 0, 0))
    );

    scalarList feedValues(feedCells.size(), feedPressure);
    boolList fullFilm(mesh.nCells(), true);
    if (gMin(thetaFill.primitiveField()) < 1 - 1e-8)
    {
        forAll(fullFilm, celli)
        {
            fullFilm[celli] = p[celli] > policyTolerance;
        }
    }
    forAll(feedCells, i)
    {
        fullFilm[feedCells[i]] = true;
        p[feedCells[i]] = feedPressure;
        thetaFill[feedCells[i]] = 1;
    }

    const label minimumSteps = mag(omega) < small ? 2 : nTheta;
    const label maximumSteps =
        mag(omega) < small
      ? maxZeroSpeedSteps
      : maxRevolutions*nTheta;
    label consecutive = 0;
    scalar pressureError = great;
    scalar fillError = great;
    label maxPolicyIterationsUsed = 0;
    bool converged = false;

    Info<< "rpm=" << rpm
        << " deltaT=" << deltaT
        << " feedCells=" << feedCells.size()
        << " hMin=" << gMin(filmThickness.primitiveField())
        << " hMax=" << gMax(filmThickness.primitiveField())
        << nl << endl;

    for (label step = 1; step <= maximumSteps; ++step)
    {
        ++runTime;
        const scalarField oldPressure(p.primitiveField());
        const scalarField oldFill(thetaFill.primitiveField());
        const volScalarField hTheta
        (
            IOobject
            (
                "hTheta",
                runTime.name(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            filmThickness*thetaFill
        );
        const volScalarField capacity
        (
            IOobject
            (
                "capacity",
                runTime.name(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            surfaceMetric*filmThickness/runTime.deltaT()
        );
        const volScalarField rightHandSide
        (
            IOobject
            (
                "rightHandSide",
                runTime.name(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            surfaceMetric*hTheta/runTime.deltaT()
          - fvc::div
            (
                phiCouette,
                hTheta,
                "div(phiCouette,hTheta)"
            )
        );

        fvScalarMatrix baseEquation(-fvm::laplacian(diffusion, p));
        baseEquation.source() +=
            (rightHandSide.primitiveField() - capacity.primitiveField())
           *mesh.V();

        label policyIterations = 0;
        scalarField slack(mesh.nCells(), 0);

        for
        (
            policyIterations = 1;
            policyIterations <= maxActiveIterations;
            ++policyIterations
        )
        {
            DynamicList<label> cavityCells;
            forAll(fullFilm, celli)
            {
                if (!fullFilm[celli] && !feedMask[celli])
                {
                    cavityCells.append(celli);
                }
            }

            fvScalarMatrix constrainedEquation(baseEquation);
            if (!cavityCells.empty())
            {
                constrainedEquation.setValues
                (
                    cavityCells,
                    scalarList(cavityCells.size(), scalar(0))
                );
            }
            constrainedEquation.setValues(feedCells, feedValues);
            constrainedEquation.solve();
            p.correctBoundaryConditions();

            slack = -baseEquation.residual();
            bool changed = false;
            forAll(fullFilm, celli)
            {
                if (feedMask[celli])
                {
                    continue;
                }
                const scalar criterion =
                    p[celli]
                  - slack[celli]/max(baseEquation.diag()[celli], vSmall);
                const bool nextFull = criterion > policyTolerance;
                changed = changed || nextFull != fullFilm[celli];
                fullFilm[celli] = nextFull;
            }
            if (!changed)
            {
                break;
            }
        }

        if (policyIterations > maxActiveIterations)
        {
            FatalErrorInFunction
                << "Active set did not converge at step " << step
                << exit(FatalError);
        }
        maxPolicyIterationsUsed =
            max(maxPolicyIterationsUsed, policyIterations);

        slack = -baseEquation.residual();
        forAll(thetaFill, celli)
        {
            if (feedMask[celli] || fullFilm[celli])
            {
                thetaFill[celli] = 1;
            }
            else
            {
                const scalar storedCapacity =
                    capacity[celli]*mesh.V()[celli];
                thetaFill[celli] =
                    1 - slack[celli]/storedCapacity;
            }

            if
            (
                p[celli] < -1e-5
             || thetaFill[celli] < -1e-7
             || thetaFill[celli] > 1 + 1e-7
            )
            {
                FatalErrorInFunction
                    << "JFO bounds failed in cell " << celli
                    << ": p=" << p[celli]
                    << ", theta=" << thetaFill[celli]
                    << exit(FatalError);
            }
            p[celli] = max(p[celli], scalar(0));
            thetaFill[celli] =
                min(max(thetaFill[celli], scalar(0)), scalar(1));
        }
        thetaFill.correctBoundaryConditions();

        pressureError =
            gMax(mag(p.primitiveField() - oldPressure))
           /max
            (
                max(feedPressure, gMax(p.primitiveField())),
                scalar(1)
            );
        fillError =
            gMax(mag(thetaFill.primitiveField() - oldFill));
        const bool convergedNow =
            pressureError <= convergenceTolerance
         && fillError <= convergenceTolerance;
        consecutive = convergedNow ? consecutive + 1 : 0;

        if (step == 1 || (logInterval > 0 && step % logInterval == 0))
        {
            Info<< "step=" << step
                << " pMax=" << gMax(p.primitiveField())
                << " thetaMin=" << gMin(thetaFill.primitiveField())
                << " dp=" << pressureError
                << " dTheta=" << fillError
                << " activeIterations=" << policyIterations
                << endl;
        }

        if (step >= minimumSteps && consecutive >= 5)
        {
            converged = true;
            Info<< "Converged at step " << step
                << " after "
                << (mag(omega) < small ? 0 : scalar(step)/nTheta)
                << " characteristic revolutions" << endl;
            break;
        }
    }

    runTime.writeNow();

    const scalar pseudoThickness = mesh.bounds().span().y();
    const surfaceSymmTensorField faceDiffusion
    (
        fvc::interpolate(diffusion)
    );
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    scalar feedIn = 0;

    forAll(neighbour, facei)
    {
        const label own = owner[facei];
        const label nei = neighbour[facei];
        if (feedMask[own] == feedMask[nei])
        {
            continue;
        }

        const vector normal = mesh.Sf()[facei]/mesh.magSf()[facei];
        const scalar normalDiffusion =
            normal & (faceDiffusion[facei] & normal);
        const scalar pressureFlow =
           -normalDiffusion
           *(p[nei] - p[own])
           *mesh.deltaCoeffs()[facei]
           *mesh.magSf()[facei]
           /pseudoThickness;
        const label upstream =
            phiCouette[facei] >= 0 ? own : nei;
        const scalar couetteFlow =
            phiCouette[facei]
           *filmThickness[upstream]
           *thetaFill[upstream]
           /pseudoThickness;
        const scalar outwardFromOwner = pressureFlow + couetteFlow;
        feedIn += feedMask[own] ? outwardFromOwner : -outwardFromOwner;
    }

    const auto axialOutflow =
    [&]
    (
        const word& patchName
    )
    {
        const label patchi =
            mesh.boundary().findIndex(patchName);
        if (patchi < 0)
        {
            FatalErrorInFunction
                << "Missing patch " << patchName
                << exit(FatalError);
        }

        const fvPatch& patch = mesh.boundary()[patchi];
        const fvPatchScalarField& patchPressure =
            p.boundaryField()[patchi];
        const fvPatchSymmTensorField& patchDiffusion =
            diffusion.boundaryField()[patchi];
        const tmp<scalarField> tSnGrad = patchPressure.snGrad();
        const scalarField& snGrad = tSnGrad();
        scalar outflow = 0;

        forAll(patch, facei)
        {
            const vector normal =
                patch.Sf()[facei]/patch.magSf()[facei];
            const scalar normalDiffusion =
                normal & (patchDiffusion[facei] & normal);
            outflow +=
               -normalDiffusion
               *snGrad[facei]
               *patch.magSf()[facei]
               /pseudoThickness;
        }
        return outflow;
    };

    scalar axialZ0Out = axialOutflow("axialEndZ0");
    scalar axialZLOut = axialOutflow("axialEndZL");
    reduce(feedIn, sumOp<scalar>());
    reduce(axialZ0Out, sumOp<scalar>());
    reduce(axialZLOut, sumOp<scalar>());
    const scalar netOut = axialZ0Out + axialZLOut - feedIn;
    const scalar relativeImbalance =
        mag(netOut)
       /max
        (
            max(mag(feedIn), mag(axialZ0Out) + mag(axialZLOut)),
            vSmall
        );

    const volVectorField pressureGradient(fvc::grad(p));
    vector pressureForce(vector::zero);
    vector viscousForce(vector::zero);
    scalar journalTorque = 0;
    scalar totalArea = 0;
    scalar filledArea = 0;
    scalar cavityArea = 0;

    forAll(p, celli)
    {
        const scalar theta = centres[celli].x()/meanRadius.value();
        const scalar z = centres[celli].z();
        const scalar journalRadius =
            meanRadius.value()
          + (0.5*length.value() - z)*Foam::tan(gamma);
        const scalar q =
            ex*Foam::sin(theta) - ey*Foam::cos(theta);
        const scalar journalRayRadius =
            q
          + Foam::sqrt
            (
                sqr(journalRadius)
              - sqr(eccentricity.value())
              + sqr(q)
            );
        const scalar radialX =
            (journalRayRadius*Foam::sin(theta) - ex)/journalRadius;
        const scalar radialY =
            (-journalRayRadius*Foam::cos(theta) - ey)/journalRadius;
        const vector normal
        (
            cosGamma*radialX,
            cosGamma*radialY,
            Foam::sin(gamma)
        );
        const vector tangent(-radialY, radialX, 0);
        const scalar journalArea =
            journalRadius
           *mesh.V()[celli]
           /(meanRadius.value()*cosGamma*pseudoThickness);
        const scalar pressureGauge =
            cavitationPressure.value()
          + p[celli]
          - ambientPressure.value();
        const scalar circumferentialPressureGradient =
            pressureGradient[celli].x()
           *meanRadius.value()
           /journalRadius;
        const scalar shear =
            thetaFill[celli]
           *(
               -viscosity.value()
               *omega
               *journalRadius
               /filmThickness[celli]
               -0.5
               *filmThickness[celli]
               *circumferentialPressureGradient
            );
        const scalar midSurfaceArea =
            surfaceMetric[celli]*mesh.V()[celli]/pseudoThickness;

        pressureForce -= pressureGauge*normal*journalArea;
        viscousForce += shear*tangent*journalArea;
        journalTorque += shear*journalRadius*journalArea;
        totalArea += midSurfaceArea;
        filledArea += thetaFill[celli]*midSurfaceArea;
        if (thetaFill[celli] < 1 - 1e-8)
        {
            cavityArea += midSurfaceArea;
        }
    }

    reduce(pressureForce, sumOp<vector>());
    reduce(viscousForce, sumOp<vector>());
    reduce(journalTorque, sumOp<scalar>());
    reduce(totalArea, sumOp<scalar>());
    reduce(filledArea, sumOp<scalar>());
    reduce(cavityArea, sumOp<scalar>());

    const scalar absoluteMinimum =
        cavitationPressure.value()
      + min(gMin(p.primitiveField()), endPressure);
    const scalar absoluteMaximum =
        cavitationPressure.value() + gMax(p.primitiveField());
    const bool accepted =
        converged
     && absoluteMinimum >= cavitationPressure.value() - 1e-6
     && relativeImbalance <= 0.005;

    Info<< nl
        << "JFO_RESULT"
        << " converged=" << converged
        << " accepted=" << accepted
        << " rpm=" << rpm
        << " pAbsMin=" << absoluteMinimum
        << " pAbsMax=" << absoluteMaximum
        << " pAbsMaxOverFeedGauge="
        << absoluteMaximum/feedGaugePressure.value()
        << " pGaugeMaxOverFeedGauge="
        << (
               absoluteMaximum - ambientPressure.value()
           )/feedGaugePressure.value()
        << " thetaMin=" << gMin(thetaFill.primitiveField())
        << " thetaMean=" << filledArea/totalArea
        << " cavityAreaFraction=" << cavityArea/totalArea
        << " feedIn=" << feedIn
        << " axialZ0Out=" << axialZ0Out
        << " axialZLOut=" << axialZLOut
        << " netOut=" << netOut
        << " relativeImbalance=" << relativeImbalance
        << " pressureForce=" << pressureForce
        << " viscousForce=" << viscousForce
        << " totalForce=" << pressureForce + viscousForce
        << " journalTorque=" << journalTorque
        << " pressureError=" << pressureError
        << " fillError=" << fillError
        << " maxActiveIterations=" << maxPolicyIterationsUsed
        << nl << endl;

    return accepted ? 0 : 2;
}

// ************************************************************************* //
