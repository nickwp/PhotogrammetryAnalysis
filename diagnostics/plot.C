#include <stdlib.h>     /* strtol */


void getAsymmRMS(TH1F *h_in, double center, double &RMSleft, double &RMSright)
{
    TH1F *htmp = (TH1F*)h_in->Clone();
    htmp->Scale(1./htmp->Integral());
    
    int binCenter = htmp->FindBin(center);
    double centerBinWidth = htmp->GetBinWidth(binCenter);
    
    double right_integral=htmp->GetBinContent(binCenter)/2;
    double add_to_left_side = 0;
    int ibin;
    for (ibin=htmp->GetNbinsX(); ibin>=1; ibin--) {
        double val = htmp->GetBinContent(ibin);
        
        right_integral += val;
        
        if (right_integral>=0.1587) break;
    }
    RMSright = fabs(htmp->GetBinLowEdge(ibin+1)-center);
    
    double left_integral=htmp->GetBinContent(binCenter)/2;// - add_to_left_side;
    for (ibin=1; ibin<=htmp->GetNbinsX(); ibin++) {
        left_integral += htmp->GetBinContent(ibin);
        if (left_integral>=0.1587) break;
    }
    RMSleft = fabs(center-htmp->GetBinLowEdge(ibin));
    
}

TH2F *GetLogTH1(TString hName, TString hTitle, Int_t nBinsX, Double_t xMin, Double_t xMax, Int_t nBinsY, Double_t yMin, Double_t yMax, Int_t nBinsZ=0, Double_t zMin=0, Double_t zMax=0) {

  const Int_t xlnbins = nBinsX;
  Double_t lxmin = xMin;
  Double_t lxmax = xMax;
  Double_t logxmin = log10(lxmin);
  Double_t logxmax = log10(lxmax);
  Double_t binwidthx = (logxmax-logxmin)/xlnbins;
  Double_t xbins[xlnbins+1];
  xbins[0] = lxmin;
  for (Int_t i=1;i<=xlnbins;i++) {
    xbins[i] = TMath::Power(10,logxmin+i*binwidthx);
  }

  //Double_t ybins[nBinsY+1];
  //ybins[0] = yMin;
  //double yBinWidth = (yMax-yMin)/nBinsY;
  //for (int iy=1; iy<=nBinsY; iy++) {
  //  ybins[iy] = yMin + iy*yBinWidth;
  //} 
  //
  //Double_t zbins[nBinsZ+1];
  //zbins[0] = zMin;
  //double zBinWidth = (zMax-zMin)/nBinsZ;
  //for (int iz=1; iz<=nBinsZ; iz++) {
  //  zbins[iz] = zMin + iz*zBinWidth;
  //}	

  TH2F *hlog;
  
  //if (nBinsZ<0 && nBinsY<0) hlog = new TH1F(hName,hTitle,xlnbins,xbins);
  //else if (nBinsZ<0 && nBinsY>0) hlog = new TH2F(hName,hTitle,xlnbins,xbins, nBinsY, ybins);
  //else if (nBinsZ<0 && nBinsY>0)
  hlog = new TH2F(hName,hTitle, nBinsY, yMin, yMax, xlnbins, xbins);
  //else hlog = new TH2F(hName,hTitle,xlnbins,xbins, nBinsY, ybins, nBinsZ, zbins);

  //hlog->Sumw2();

  return hlog;
}


void plot(){

  gROOT->ProcessLine(Form(".! mkdir -p images"));

  TTree t_photogram("t_photogram", "Photogrammetry Results");
  //t_photogram.ReadFile("../results/SK_demo1_features.txt");
  //const int nMaxImages = 4;
  //t_photogram.ReadFile("../results/SK_demo2_features.txt");
  //const int nMaxImages = 15;
  t_photogram.ReadFile("../results/SK_demo3_features.txt");
  const int nMaxImages = 11;

  Char_t          FeatureID[9];
  Int_t           nImages;
  Int_t           ImagePosition[nMaxImages][2];
  Double_t        ExpectedWorldPosition[3];
  Double_t        RecoWorldPosition[3];
  Double_t        ReprojectedPosition[nMaxImages][2];
  Double_t        ReprojectionError[nMaxImages];
  
  t_photogram.SetBranchAddress("FeatureID", FeatureID);
  t_photogram.SetBranchAddress("nImages", &nImages);
  t_photogram.SetBranchAddress("ImagePosition", ImagePosition);
  t_photogram.SetBranchAddress("ExpectedWorldPosition", ExpectedWorldPosition);
  t_photogram.SetBranchAddress("RecoWorldPosition", RecoWorldPosition);
  t_photogram.SetBranchAddress("ReprojectedPosition", ReprojectedPosition);
  t_photogram.SetBranchAddress("ReprojectionError", ReprojectionError);

  TCanvas *c_nimages = new TCanvas(1);
  t_photogram.Draw("nImages");
  TString s_title = Form("Total Number of Features = %lld; Number of Matched Images; Number of Features", t_photogram.GetEntries());
  TH1D *htemp = (TH1D*)gPad->GetPrimitive("htemp");
  htemp->SetTitle(s_title);
  htemp->SetLineWidth(3);

  c_nimages->Print("images/nImages.png");
  
  TH1F *h_ReprojectionError[nMaxImages];
  TH1F *h_ReprojectionErrorAvg[nMaxImages];
  
  //TH2F *h_ReprojectionErrorSpatial = GetLogTH1("h_ReprojectionErrorSpatial", "Photo Pixel Space; Distance from Center (pixels); Reprojection Error (pixels); Number of Features*Images", 20, 0.05, 10, 25, 0, 1500);
  TH2F *h_ReprojectionErrorSpatial = GetLogTH1("h_ReprojectionErrorSpatial", "Photo Pixel Space; Distance from Center (pixels); Reprojection Error (pixels); Number of Features*Images", 20, 0.05, 12.5, 25, 0, 2500);
  
  TGraph2D *g_ReprojectionError[nMaxImages];
  int nGraphPoints[nMaxImages] = {0};
  
  const int nFeatureSets = 2;
  TH1F *h_ReprojectionErrorAvg_Features[nFeatureSets];
  
  TCanvas *c_ReprojectionError = new TCanvas(1);
  //c_ReprojectionError->SetLogy(1);
  t_photogram.Draw("ReprojectionError","ReprojectionError>0");

  htemp = (TH1D*)gPad->GetPrimitive("htemp");
  htemp->SetTitle(";Reprojection Error (pixels); Number of Features*Images");

  h_ReprojectionError[0] = (TH1F*)htemp->Clone();
  
  for (int iimg=0; iimg<nMaxImages; iimg++) {
    g_ReprojectionError[iimg] = new TGraph2D();
    g_ReprojectionError[iimg]->SetName(Form("g_ReprojectionError_%d", iimg));
    
    if (iimg) {
      h_ReprojectionError[iimg] = (TH1F*)h_ReprojectionError[0]->Clone();
      h_ReprojectionError[iimg]->Reset();
    }
    h_ReprojectionError[iimg]->SetName(Form("h_ReprojectionError_nimg%d",iimg));

    h_ReprojectionErrorAvg[iimg] = (TH1F*)h_ReprojectionError[0]->Clone();
    h_ReprojectionErrorAvg[iimg]->Reset();
    h_ReprojectionErrorAvg[iimg]->SetName(Form("h_ReprojectionErrorAvg_nimg%d",iimg));
    
  }

  for (int iset=0; iset<nFeatureSets; iset++) {
    h_ReprojectionErrorAvg_Features[iset] = (TH1F*)h_ReprojectionError[0]->Clone();
    h_ReprojectionErrorAvg_Features[iset]->Reset();
    h_ReprojectionErrorAvg_Features[iset]->SetName(Form("h_ReprojectionErrorAvg_Features_set%d",iset));
  }

  
  
  for (int ientry=0; ientry<t_photogram.GetEntries(); ientry++) {

    t_photogram.GetEntry(ientry);

    char * pEnd;
    long int PMT_ID = strtol (FeatureID, &pEnd, 10);
    long int Feature_ID = - strtol (pEnd, &pEnd, 10);
    //cout << FeatureID << " " << PMT_ID << " " << Feature_ID << " " << (Feature_ID-1)/8+1 << endl;

    double AvgError = 0;
    for (int iimg=0; iimg<nMaxImages; iimg++) {

      if (ReprojectionError[iimg]<=0) continue;

      h_ReprojectionError[nImages-1]->Fill(ReprojectionError[iimg]);

      float x_pixel = ImagePosition[iimg][0] - 2000;
      float y_pixel = ImagePosition[iimg][1] - 1500;
      float radius_pixel = sqrt(x_pixel*x_pixel + y_pixel*y_pixel);
      h_ReprojectionErrorSpatial->Fill(radius_pixel, ReprojectionError[iimg]);
      
      g_ReprojectionError[iimg]->SetPoint(nGraphPoints[iimg], ImagePosition[iimg][0], -ImagePosition[iimg][1], ReprojectionError[iimg]);
      nGraphPoints[iimg]++;
      
      AvgError += ReprojectionError[iimg];

    }
    
    AvgError = AvgError/nImages;
    h_ReprojectionErrorAvg[0]->Fill(AvgError);
    h_ReprojectionErrorAvg[nImages-1]->Fill(AvgError);

    if (!Feature_ID) // Dynode/reflection
      h_ReprojectionErrorAvg_Features[Feature_ID]->Fill(AvgError);
    else { // Bolts
      //int feature_idx = (Feature_ID-1)/6+1;
      int feature_idx = 1;
      h_ReprojectionErrorAvg_Features[feature_idx]->Fill(AvgError);
    }
    
  }
  

  // Raw ReprojectionError
  c_ReprojectionError->cd();

  TLegend *leg_reproj = new TLegend(0.5, 0.2, 0.9, 0.9);
  leg_reproj->SetFillColor(0);
  leg_reproj->SetHeader("Matched Images (Mean)");

  TGraphAsymmErrors *g_mean_errors = new TGraphAsymmErrors();
  g_mean_errors->SetName("g_mean_errors");
  int nMeanErrors = 0;

  for (int iimg=0; iimg<nMaxImages; iimg++) {

    if (!h_ReprojectionError[iimg]->GetEntries()) continue;
    
    h_ReprojectionError[iimg]->SetLineWidth(3);
    if (!iimg) h_ReprojectionError[iimg]->SetLineWidth(5);
    h_ReprojectionError[iimg]->SetLineColor(iimg-1 + (!iimg)*2);

    h_ReprojectionError[iimg]->Draw("same");

    float MeanError = h_ReprojectionError[iimg]->GetMean();
    double rms_left, rms_right;
    rms_left = h_ReprojectionError[iimg]->GetRMS();
    rms_right = h_ReprojectionError[iimg]->GetRMS();
    //getAsymmRMS(h_ReprojectionError[iimg], MeanError, rms_left, rms_right);

    if (iimg) {
      g_mean_errors->SetPoint(nMeanErrors, iimg+1, MeanError);
      g_mean_errors->SetPointError(nMeanErrors, 0, 0, rms_left, rms_right);
      nMeanErrors++;
    }
    
    TString leg_entry = Form("%d", iimg+1);
    if (!iimg) leg_entry = "Total";
    leg_entry += Form(" (%.2f pixels)", MeanError);
    leg_reproj->AddEntry(h_ReprojectionError[iimg], leg_entry, "l");

  }
  leg_reproj->Draw();

  c_ReprojectionError->Print("images/ReprojectionError.png");

  TCanvas *c_MeanError = new TCanvas(1);
  g_mean_errors->Draw("AP");
  g_mean_errors->SetMarkerSize(1);
  g_mean_errors->GetXaxis()->SetTitle("Number of Matched Images");
  g_mean_errors->GetYaxis()->SetTitle("Mean Reprojection Error (pixels)");
  c_MeanError->Print("images/MeanError_vs_nImg.png");
  
  // Average ReprojectionError
  TCanvas *c_ReprojectionErrorAvg = new TCanvas(1);
  TLegend *leg_reproj_avg = new TLegend(0.5, 0.2, 0.9, 0.9);
  leg_reproj_avg->SetFillColor(0);
  leg_reproj_avg->SetHeader("Matched Images (Mean)");

  
  for (int iimg=0; iimg<nMaxImages; iimg++) {

    if (!h_ReprojectionErrorAvg[iimg]->GetEntries()) continue;

    h_ReprojectionErrorAvg[iimg]->SetTitle(";Average Reprojection Error (pixels); Number of Features");
    h_ReprojectionErrorAvg[iimg]->SetLineWidth(3);
    if (!iimg) h_ReprojectionErrorAvg[iimg]->SetLineWidth(5);
    h_ReprojectionErrorAvg[iimg]->SetLineColor(iimg-1 + (!iimg)*2);

    if (!iimg)
      h_ReprojectionErrorAvg[iimg]->Draw();
    else 
      h_ReprojectionErrorAvg[iimg]->Draw("same");

    float MeanError = h_ReprojectionErrorAvg[iimg]->GetMean();
    
    TString leg_entry = Form("%d", iimg+1);
    if (!iimg) leg_entry = "Total";
    leg_entry += Form(" (%.2f pixels)", MeanError);
    
    leg_reproj_avg->AddEntry(h_ReprojectionErrorAvg[iimg], leg_entry, "l");
  }
  leg_reproj_avg->Draw();

  c_ReprojectionErrorAvg->Print("images/ReprojectionErrorAvg.png");

  
  // Feature-separated ReprojectionError
  TCanvas *c_ReprojectionErrorFeatures = new TCanvas(1);
  TLegend *leg_reproj_features = new TLegend(0.5, 0.45, 0.9, 0.9);
  leg_reproj_features->SetFillColor(0);
  leg_reproj_features->SetHeader("Feature Set (Mean)");

  for (int iset=0; iset<nFeatureSets; iset++) {

    h_ReprojectionErrorAvg_Features[iset]->SetTitle(";Average Reprojection Error (pixels); Number of Features");
    h_ReprojectionErrorAvg_Features[iset]->SetLineWidth(3);
    if (!iset) h_ReprojectionErrorAvg_Features[iset]->SetLineWidth(5);
    h_ReprojectionErrorAvg_Features[iset]->SetLineColor(iset+2);

    if (!iset)
      h_ReprojectionErrorAvg[0]->Draw();
    
    h_ReprojectionErrorAvg_Features[iset]->Draw("same");

    //TString leg_entry = Form("Bolt set %d", iset);
    TString leg_entry = "Bolts";
    if (!iset) {
      leg_entry = "Total";
      leg_entry += Form(" (%.2f pixels)", h_ReprojectionErrorAvg[0]->GetMean());
      leg_reproj_features->AddEntry(h_ReprojectionErrorAvg[0], leg_entry, "l");

      leg_entry = "Dynode/Reflection";
    }

    leg_entry += Form(" (%.2f pixels)", h_ReprojectionErrorAvg_Features[iset]->GetMean());
    leg_reproj_features->AddEntry(h_ReprojectionErrorAvg_Features[iset], leg_entry, "l");

  }
  leg_reproj_features->Draw();
  c_ReprojectionErrorFeatures->Print("images/ReprojectionErrorFeatures.png");

  // In image space
  TCanvas *c_imagespace = new TCanvas(1);
  //TH3F *frame3d = new TH3F("frame3d","Reprojection Errors in All Images; x (pixel); y (pixel)",10,0,4000,10,0,-3000,10,0,4);
  TH3F *frame3d = new TH3F("frame3d","Reprojection Errors in All Images",10,0,4000,10,0,-3000,10,0,4);
  frame3d->Draw();
  frame3d->GetYaxis()->SetLabelSize(0);
  frame3d->GetXaxis()->SetLabelSize(0);
  
  for (int iimg=0; iimg<nMaxImages; iimg++) {
    g_ReprojectionError[iimg]->SetMarkerStyle(20);
    g_ReprojectionError[iimg]->SetMarkerSize(.2);

    g_ReprojectionError[iimg]->Draw("PCOL SAME");  
  }
  c_imagespace->SetPhi(0);
  c_imagespace->SetTheta(90);
  c_imagespace->Print("images/imagespace.png");

  gStyle->SetPadRightMargin(0.15);
  
  TCanvas *c_error_vs_r = new TCanvas(1);
  c_error_vs_r->SetLogy(1);

  //for (int jbin=1; jbin<=h_ReprojectionErrorSpatial->GetNbinsY(); jbin++) {
  //  float bin_width = h_ReprojectionErrorSpatial->GetYaxis()->GetBinWidth(jbin);
  //
  //  for (int ibin=1; ibin<=h_ReprojectionErrorSpatial->GetNbinsX(); ibin++)
  //    h_ReprojectionErrorSpatial->SetBinContent(ibin, jbin, h_ReprojectionErrorSpatial->GetBinContent(ibin, jbin)/bin_width);
  //}
  
  h_ReprojectionErrorSpatial->Draw("COLZ");
  TProfile *h_SpatialMean = h_ReprojectionErrorSpatial->ProfileX();
  h_SpatialMean->SetMarkerColor(kGray);
  h_SpatialMean->SetLineColor(kGray);
  h_SpatialMean->SetLineWidth(2);
  h_SpatialMean->Draw("same");
  
  h_ReprojectionErrorSpatial->GetZaxis()->SetTitleOffset(0.8);
  c_error_vs_r->Print("images/ReprojectionErrorSpatial.png");
}
