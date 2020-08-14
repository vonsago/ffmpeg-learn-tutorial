//
// Created by von on 2020/8/1.
//

#include <stdio.h>
#include <string.h>


/*
 * FLV struct
 */
typedef struct
{
    byte Signture[3];
    byte Version;
    byte Flags;
    uint DataOffset; //file head size (UI32)
} FLV_HEADER;

typedef struct
{
    byte data[8];
} SC_NUM;

typedef struct
{
    byte TagType;
    byte DataSize[3];
    byte Timestamp[3];
    uint Reserved;
} TAG_HEADER;

/*
 * TS struct
 */
typedef struct PMT_CAT_INFO_T {
    unsigned int uiPMT_CA_system_id: 16;
    unsigned int uiPMT_CA_PID: 13;
} PMT_CAT_INFO_T;

typedef struct PMT_INFO_T {
    unsigned int uiProgramNumber: 16;
    unsigned int uiVideoPID: 13;
    unsigned int uiAudioPID[PMT_VUDIO_COUNT];
} PMT_INFO_T;

typedef struct CAT_INFO_T {
    unsigned int uiCA_system_id: 16;
    unsigned int uiCA_PID: 13;
} CAT_INFO_T;

typedef struct PAT_INFO_T {
    unsigned int uiProgramNumber: 16;
    unsigned int uiPMT_PID: 13;
} PAT_INFO_T;

typedef struct TS_PMT_STREAM_T {
    unsigned int uiStream_type: 8;
    unsigned int uiReserved_fifth: 3;
    unsigned int uiElementary_PID: 13;
    unsigned int uiReserved_sixth: 4;
    unsigned int uiES_info_length: 12;
    unsigned char aucDescriptor[PMT_DESCRIPTOR_MAX];
} TS_PMT_STREAM_T;

typedef struct TS_PMT_T {
    unsigned int uiTable_id: 8;
    unsigned int uiSection_syntax_indicator: 1;
    unsigned int uiZero: 1;
    unsigned int uiReserved_first: 2;
    unsigned int uiSection_length: 12;
    unsigned int uiProgram_number: 16;
    unsigned int uiReserved_second: 2;
    unsigned int uiVersion_number: 5;
    unsigned int uiCurrent_next_indicator: 1;
    unsigned int uiSection_number: 8;
    unsigned int uiLast_section_number: 8;
    unsigned int uiReserved_third: 3;
    unsigned int uiPCR_PID: 13;
    unsigned int uiReserved_fourth: 4;
    unsigned int uiProgram_info_length: 12;
    unsigned char aucProgramDescriptor[PMT_PROGRAM_DESCRIPTOR_MAX];
    TS_PMT_STREAM_T stPMT_Stream[PMT_DESCRIPTOR_MAX];
    unsigned long uiCRC_32: 32;
} TS_PMT_T;

typedef struct TS_PAT_PROGRAM_T {
    unsigned int uiProgram_number: 16;
    unsigned int uiReserved_third: 3;
    unsigned int uiProgram_map_PID: 13;
} TS_PAT_PROGRAM_T;

typedef struct TS_PAT_T {
    unsigned int uiTable_id: 8;
    unsigned int uiSection_syntax_indicator: 1;
    unsigned int uiZero: 1;
    unsigned int uiReserved_first: 2;
    unsigned int uiSection_length: 12;
    unsigned int uiTransport_stream_id: 16;
    unsigned int uiReserved_second: 2;
    unsigned int uiVersion_number: 5;
    unsigned int uiCurrent_next_indicator: 1;
    unsigned int uiSection_number: 8;
    unsigned int uiLast_section_number: 8;

    TS_PAT_PROGRAM_T stPAT_Program[PAT_PROGARM_MAX];

    unsigned int uiNetwork_PID: 13;
    unsigned long uiCRC_32: 32;
} TS_PAT_T;

typedef struct SECTION_HEAD_T {
    unsigned int uiTable_id: 8;
    unsigned int uiSection_syntax_indicator: 1;
    unsigned int uiZero: 1;
    unsigned int uiReservedFirst: 2;
    unsigned int uiSection_Length: 12;
    unsigned int uiTransport_stream_id: 16;
    unsigned int uiReservedSecond: 2;
    unsigned int uiVersion_number: 5;
    unsigned int uiCurrent_next_indicator: 1;
    unsigned int uiSection_number: 8;
    unsigned int uiLast_section_number: 8;
} SECTION_HEAD_T;

typedef struct TS_PACKAGE_HEAD_T {
    unsigned int uiSync_byte: 8;
    unsigned int uiTransport_error_indicator: 1;
    unsigned int uiPayload_unit_start_indicator: 1;
    unsigned int uiTransport_priority: 1;
    unsigned int uiPID: 13;
    unsigned int uiTransport_scrambling_control: 2;
    unsigned int uiAdapptation_field_control: 2;
    unsigned int uiContinuity_counter: 4;
} TS_PACKAGE_HEAD_T;

int JudgmentPackageTenTimes(FILE *pfTsFile, int iTsPosition, int iTsLength) {
    int first_char = 0;
    if (-1 == fseek(pfTsFile, iTsPosition + 1, SEEK_SET) || -1 == fseek(pfTsFile, iTsLength - 1, SEEK_CUR) || feof(pfTsFile)) {
        return -1;
    }
    first_char = fgetc(pfTsFile);
    if (0x47 != first_char) {
        return -1;
    }
    return iTsLength;
}

int GetTsLength(FILE *pfTsFile, int *piTsPosition) {
    int iFirstPackageByte = 0;
    while (!feof(pfTsFile)) {
        iFirstPackageByte = fgetc(pfTsFile);
        // find sync sign
        if (0x47 == iFirstPackageByte) {
            // packet length 188
            if (188 == JudgmentPackageTenTimes(pfTsFile, *piTsPosition, 188)) {
                return 188;
            }
            // packet length 204
            if (204 == JudgmentPackageTenTimes(pfTsFile, *piTsPosition, 204)) {
                return 204;
            }
        }
        (*piTsPosition)++;

        if (-1 == fseek(pfTsFile, *piTsPosition, SEEK_SET)) {
            printf("The file error\n");
            return -1;
        }
    }
    printf("The file is not the transport stream\n");
    return -1;
}

int main(int argc, char **argvs) {
    // argvs[0]
    int iPosT = 0;
    char *file = argvs[1];
    fprintf(stdout, "Trans TS file to FLV: %s\n", file);
    FILE *ifile = fopen(file, "rb+");
    if (ifile == NULL) {
        fprintf(stdout, "Failed to open files!\n");
        return -1;
    }

    int ts_len = GetTsLength(ifile, &iPosT);
    if (iPosT != 0) {
        fprintf(stderr, "Ts file is not valid");
        return -1;
    }
    fprintf(stdout, "Ts header Length is %d\n", ts_len);

    PAT_INFO_T stPAT_Info[PROGRAM_MAX] = {0};
    PMT_INFO_T stPMT_Info[PROGRAM_MAX] = {0};
    CAT_INFO_T stCAT_Info[CA_SYSTEM_MAX] = {0};
    // parse PAT
    int iProgramCount = ParsePAT_Table(ifile, iPosT, ts_len, stPAT_Info);
    printf("Program count is %d\n", iProgramCount);

    ParseAllProgramPMT(ifile, iPosT, ts_len, stPAT_Info, iProgramCount, stPMT_Info);
    return 0;
}