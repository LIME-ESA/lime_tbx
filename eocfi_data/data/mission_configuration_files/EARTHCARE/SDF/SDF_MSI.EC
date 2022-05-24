<?xml version = "1.0"?>
<Earth_Explorer_File xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://eop-cfi.esa.int/CFI http://eop-cfi.esa.int/CFI/EE_CFI_SCHEMAS/EO_OPER_MPL_SW_DEF_0303.XSD"
  xmlns="http://eop-cfi.esa.int/CFI" schemaVersion="3.3">
  <Earth_Explorer_Header>
    <Fixed_Header>
      <File_Name>SDF_MSI</File_Name>
      <File_Description>Swath Definition File</File_Description>
      <Notes>Local Normal Pointing + Yaw Steering Attitude (AOCS mode 2)</Notes>
      <Mission>EarthCARE</Mission>
      <File_Class>TEST</File_Class>
      <File_Type>MPL_SW_DEF</File_Type>
      <Validity_Period>
        <Validity_Start>UTC=0000-00-00T00:00:00</Validity_Start>
        <Validity_Stop>UTC=9999-99-99T99:99:99</Validity_Stop>
      </Validity_Period>
      <File_Version>0001</File_Version>
      <Source>
        <System>Manual</System>
        <Creator>ESA/ESRIN</Creator>
        <Creator_Version>1.0</Creator_Version>
        <Creation_Date>UTC=2017-07-20T12:00:00</Creation_Date>
      </Source>
    </Fixed_Header>
    <Variable_Header></Variable_Header>
  </Earth_Explorer_Header>
  <Data_Block type="xml">
      <Swath>
        <Output_File_Description>MSI</Output_File_Description>
        <Output_File_Type>MPL_SWTREF</Output_File_Type>
        <Swath_Type>open</Swath_Type>
        <Num_Swath_Records>1200</Num_Swath_Records>
        <Refraction>
          <Model>NO_REF</Model>
          <Freq unit="MHz">0000000000</Freq>
        </Refraction>
         <List_of_Swath_Points count="3">
          <Swath_Point>
            <Pointing_Geometry>
              <Azimuth unit="deg">+270.000000</Azimuth>
              <Elevation unit="deg">+078.700000</Elevation>
              <Altitude unit="m">000000.000</Altitude>
            </Pointing_Geometry>
          </Swath_Point>
          <Swath_Point>
            <Pointing_Geometry>
              <Azimuth unit="deg">+090.000000</Azimuth>
              <Elevation unit="deg">+090.000000</Elevation>
              <Altitude unit="m">000000.000</Altitude>
            </Pointing_Geometry>
          </Swath_Point>
          <Swath_Point>
            <Pointing_Geometry>
              <Azimuth unit="deg">+090.000000</Azimuth>
              <Elevation unit="deg">+078.700000</Elevation>
              <Altitude unit="m">000000.000</Altitude>
            </Pointing_Geometry>
          </Swath_Point>
        </List_of_Swath_Points>
        <Sat_Nominal_Att>
          <AOCS_Model>2</AOCS_Model>
        </Sat_Nominal_Att>
        <Sat_Att>
          <Angle_Model>
            <Angle_1 unit="deg">+000.000000</Angle_1>
            <Angle_2 unit="deg">+005.637000</Angle_2>
            <Angle_3 unit="deg">+000.000000</Angle_3>
          </Angle_Model>
        </Sat_Att>
        <Instr_Att>
          <None></None>
        </Instr_Att>
      </Swath>
  </Data_Block>
</Earth_Explorer_File>
